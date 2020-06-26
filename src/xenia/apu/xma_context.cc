/**
******************************************************************************
* Xenia : Xbox 360 Emulator Research Project                                 *
******************************************************************************
* Copyright 2015 Ben Vanik. All rights reserved.                             *
* Released under the BSD license - see LICENSE in the root for more details. *
******************************************************************************
*/

#include "xenia/apu/xma_context.h"

#include <algorithm>
#include <cstring>

#include "xenia/apu/xma_decoder.h"
#include "xenia/apu/xma_helpers.h"
#include "xenia/base/bit_stream.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/base/ring_buffer.h"

extern "C" {
#pragma warning(push)
#pragma warning(disable : 4101 4244 5033)
#include "third_party/FFmpeg/libavcodec/avcodec.h"
#include "third_party/FFmpeg/libavcodec/wma.h"
#pragma warning(pop)
}  // extern "C"

// Credits for most of this code goes to:
// https://github.com/koolkdev/libertyv/blob/master/libav_wrapper/xma2dec.c

namespace xe {
namespace apu {

XmaContext::XmaContext() = default;

XmaContext::~XmaContext() {
  if (context_) {
    if (avcodec_is_open(context_)) {
      avcodec_close(context_);
    }
    av_free(context_);
  }
  if (decoded_frame_) {
    av_frame_free(&decoded_frame_);
  }
  //if (current_frame_) {
  //  delete[] current_frame_;
 // }
}

int XmaContext::Setup(uint32_t id, Memory* memory, uint32_t guest_ptr) {
  id_ = id;
  memory_ = memory;
  guest_ptr_ = guest_ptr;

  // Allocate ffmpeg stuff:
  packet_ = av_packet_alloc();
  assert_not_null(packet_);

  // find the XMA2 audio decoder
  codec_ = avcodec_find_decoder(AV_CODEC_ID_XMA2);
  if (!codec_) {
    XELOGE("XmaContext {}: Codec not found.", id);
    return 1;
  }

  parser_ = av_parser_init(codec_->id);
  if (!parser_) {
    XELOGE("XmaContext {}: Parser failed to initalize.", id);
    return 1;
  }

  context_ = avcodec_alloc_context3(codec_);
  if (!context_) {
    XELOGE("XmaContext {}: Couldn't allocate context.", id);
    return 1;
  }

  // This is automatically freed with the context.
  context_->extradata = static_cast<uint8_t*>(av_malloc(sizeof(Xma2ExtraData)));
  if (!context_->extradata) {
    XELOGE("XmaContext {}: Couldn't allocate extra data.", id);
    return 1;
  }
  std::memset(context_->extradata, 0, sizeof(Xma2ExtraData));
  context_->extradata_size = sizeof(Xma2ExtraData);

  // Initialize these to 0. They'll actually be set later.
  context_->channels = 0;
  context_->sample_rate = 0;

  decoded_frame_ = av_frame_alloc();
  if (!decoded_frame_) {
    XELOGE("XmaContext {}: Couldn't allocate frame.", id);
    return 1;
  }

  // Current frame stuff whatever
  // samples per frame * 2 max channels * output bytes
  //current_frame_ = new uint8_t[kSamplesPerFrame * kBytesPerSample * 2];
  //current_frame_ = new uint8_t[20 * kSamplesPerFrame * kBytesPerSample * 2];

  // FYI: We're purposely not opening the codec here. That is done later.
  return 0;
}

bool XmaContext::Work() {
  std::lock_guard<std::mutex> lock(lock_);
  if (!is_allocated() || !is_enabled()) {
    return false;
  }

  set_is_enabled(false);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  DecodePackets(&data);
  data.Store(context_ptr);
  return true;
}

void XmaContext::Enable() {
  std::lock_guard<std::mutex> lock(lock_);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);

  XELOGAPU("XmaContext: kicking context {} (buffer {} {}/{} bits)", id(),
           data.current_buffer, data.input_buffer_read_offset,
           (data.current_buffer == 0 ? data.input_buffer_0_packet_count
                                     : data.input_buffer_1_packet_count) *
               kBytesPerPacket * 8);

  data.Store(context_ptr);

  set_is_enabled(true);
}

bool XmaContext::Block(bool poll) {
  if (!lock_.try_lock()) {
    if (poll) {
      return false;
    }
    lock_.lock();
  }
  lock_.unlock();
  return true;
}

void XmaContext::Clear() {
  std::lock_guard<std::mutex> lock(lock_);
  XELOGAPU("XmaContext: reset context {}", id());

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);

  data.input_buffer_0_valid = 0;
  data.input_buffer_1_valid = 0;
  data.output_buffer_valid = 0;

  data.output_buffer_read_offset = 0;
  data.output_buffer_write_offset = 0;

  data.Store(context_ptr);
}

void XmaContext::Disable() {
  std::lock_guard<std::mutex> lock(lock_);
  XELOGAPU("XmaContext: disabling context {}", id());
  set_is_enabled(false);
}

void XmaContext::Release() {
  // Lock it in case the decoder thread is working on it now.
  std::lock_guard<std::mutex> lock(lock_);
  assert_true(is_allocated_ == true);

  set_is_allocated(false);
  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));  // Zero it.
}

int XmaContext::GetSampleRate(int id) {
  switch (id) {
    case 0:
      return 24000;
    case 1:
      return 32000;
    case 2:
      return 44100;
    case 3:
      return 48000;
  }
  assert_always();
  return 0;
}

bool XmaContext::ValidFrameOffset(uint8_t* block, size_t size_bytes,
                                  size_t frame_offset_bits) {
  uint32_t packet_num =
      GetFramePacketNumber(block, size_bytes, frame_offset_bits);
  if (packet_num == -1) {
    // Invalid packet number
    return false;
  }

  uint8_t* packet = block + (packet_num * kBytesPerPacket);
  size_t relative_offset_bits = frame_offset_bits % (kBytesPerPacket * 8);

  uint32_t first_frame_offset = xma::GetPacketFrameOffset(packet);
  if (first_frame_offset == -1 || first_frame_offset > kBytesPerPacket * 8) {
    // Packet only contains a partial frame, so no frames can start here.
    return false;
  }

  BitStream stream(packet, kBytesPerPacket * 8);
  stream.SetOffset(first_frame_offset);
  while (true) {
    if (stream.offset_bits() == relative_offset_bits) {
      return true;
    }

    if (stream.BitsRemaining() < 15) {
      // Not enough room for another frame header.
      return false;
    }

    uint64_t size = stream.Read(15);
    if ((size - 15) > stream.BitsRemaining()) {
      // Last frame.
      return false;
    } else if (size == 0x7FFF) {
      // Invalid frame (and last of this packet)
      return false;
    }

    stream.Advance(size - 16);

    // Read the trailing bit to see if frames follow
    if (stream.Read(1) == 0) {
      break;
    }
  }

  return false;
}

void XmaContext::DecodePackets(XMA_CONTEXT_DATA* data) {
  SCOPE_profile_cpu_f("apu");

  // What I see:
  // XMA outputs 2 bytes per sample
  // 512 samples per frame (128 per subframe)
  // Max output size is data.output_buffer_block_count * 256

  // This decoder is fed packets (max 4095 per buffer)
  // Packets contain "some" frames
  // 32bit header (big endian)

  // Frames are the smallest thing the SPUs can decode.
  // They can and usually will span packets.

  // Sample rates (data.sample_rate):
  // 0 - 24 kHz
  // 1 - 32 kHz
  // 2 - 44.1 kHz
  // 3 - 48 kHz

  // SPUs also support stereo decoding. (data.is_stereo)

  // Check the output buffer - we cannot decode anything else if it's
  // unavailable.
  if (!data->output_buffer_valid) {
    return;
  }

  // No available data.
  if (!data->input_buffer_0_valid && !data->input_buffer_1_valid) {
    return;
  }

  // XAudio Loops
  // loop_count:
  //  - XAUDIO2_MAX_LOOP_COUNT = 254
  //  - XAUDIO2_LOOP_INFINITE = 255
  // loop_start/loop_end are bit offsets to a specific frame

  // Translate pointers for future use.
  // Sometimes the game will use rolling input buffers. If they do, we cannot
  // assume they form a complete block! In addition, the buffers DO NOT have
  // to be contiguous!
  uint8_t* in0 = data->input_buffer_0_valid
                     ? memory()->TranslatePhysical(data->input_buffer_0_ptr)
                     : nullptr;
  uint8_t* in1 = data->input_buffer_1_valid
                     ? memory()->TranslatePhysical(data->input_buffer_1_ptr)
                     : nullptr;
  uint8_t* current_input_buffer = data->current_buffer ? in1 : in0;

  XELOGAPU("Processing context {} (offset {}, buffer {}, ptr {:p})", id(),
           data->input_buffer_read_offset, data->current_buffer,
           current_input_buffer);

  size_t input_buffer_0_size =
      data->input_buffer_0_packet_count * kBytesPerPacket;
  size_t input_buffer_1_size =
      data->input_buffer_1_packet_count * kBytesPerPacket;
  size_t input_total_size = input_buffer_0_size + input_buffer_1_size;

  size_t current_input_size =
      data->current_buffer ? input_buffer_1_size : input_buffer_0_size;
  size_t current_input_packet_count = current_input_size / kBytesPerPacket;

  // Output buffers are in raw PCM samples, 256 bytes per block.
  // Output buffer is a ring buffer. We need to write from the write offset
  // to the read offset.
  uint8_t* output_buffer = memory()->TranslatePhysical(data->output_buffer_ptr);
  uint32_t output_capacity =
      data->output_buffer_block_count * kBytesPerSubframe;
  uint32_t output_read_offset =
      data->output_buffer_read_offset * kBytesPerSubframe;
  uint32_t output_write_offset =
      data->output_buffer_write_offset * kBytesPerSubframe;

  RingBuffer output_rb(output_buffer, output_capacity);
  output_rb.set_read_offset(output_read_offset);
  output_rb.set_write_offset(output_write_offset);

  // We can only decode an entire frame and write it out at a time, so
  // don't save any samples.
  size_t output_remaining_bytes = output_rb.write_count();
  output_remaining_bytes -= data->is_stereo ? (output_remaining_bytes % 2048)
                                            : (output_remaining_bytes % 1024);

  int num_channels = data->is_stereo ? 2 : 1;

  // Try to write all data
  /*while (output_remaining_bytes > 0)  {
    if (!data->input_buffer_0_valid && !data->input_buffer_1_valid) {
      // Out of data.
      break;
    }*/
  {

    // Try to write decoded data
    if (current_frame_.size()) {
      size_t bytes_per_subframe =
          kBytesPerSample * context_->channels * kSamplesPerSubframe;
      assert_true(current_frame_.size() % bytes_per_subframe == 0);
      // size_t frames_available = current_frame_.size() / kBytesPerSample /
      //                          context_->channels / kSamplesPerFrame;
      size_t bytes_to_write = std::min(
          (output_remaining_bytes / bytes_per_subframe) * bytes_per_subframe,
          current_frame_.size());
      assert_true(output_remaining_bytes >= bytes_to_write);
      output_rb.Write(current_frame_.data(), bytes_to_write);
      std::memmove(current_frame_.data(),
                   current_frame_.data() + bytes_to_write,
                   current_frame_.size() - bytes_to_write);
      current_frame_.resize(current_frame_.size() - bytes_to_write);

      output_remaining_bytes -= bytes_to_write;
      data->output_buffer_write_offset = output_rb.write_offset() / 256;
    }

    //if (current_frame_.size()) {
    //  break;
    //}
  }

  while (!current_frame_.size()) {
    if (!data->input_buffer_0_valid && !data->input_buffer_1_valid) {
      // Out of data.
      break;
    }

    //static size_t packet_number = 0;

    size_t packet_number =
        GetFramePacketNumber(current_input_buffer, current_input_size,
                             data->input_buffer_read_offset);

    assert_true(packet_number * kBytesPerPacket < current_input_size);

    // Prepare the decoder. Reinitialize if any parameters have changed.
    PrepareDecoder(current_input_buffer, current_input_size, data->sample_rate,
                   num_channels);

    // packet_->data = current_input_buffer;
    // packet_->size = kBytesPerPacket;

    
    auto parsed = av_parser_parse2(
        parser_, context_, &packet_->data, &packet_->size, current_input_buffer + packet_number * kBytesPerPacket,
        kBytesPerPacket, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
    if (parsed <= 0) {
      XELOGE("XmaContext {}: Error parsing packet.", id());
      // TODO bail out
    }
    //packet_->data = current_input_buffer;
    //packet_->size = kBytesPerPacket;
    //int parsed = kBytesPerPacket;

    /*if (data->input_buffer_read_offset == 0)*/ {
      // Invalid offset. Go ahead and set it.
        #if 1
      uint32_t offset = xma::GetPacketFrameOffset(current_input_buffer);
      if (offset == -1) {
        // No more frames.
        if (data->current_buffer == 0) {
          data->input_buffer_0_valid = 0;
          data->input_buffer_read_offset = 0;
          data->current_buffer++;
        } else if (data->current_buffer == 1) {
          data->input_buffer_1_valid = 0;
          data->input_buffer_read_offset = 0;
          data->current_buffer--;
        }
      } else {
        //data->input_buffer_read_offset += decoded_frame_->nb_samples / 128;
        packet_number++;
        data->input_buffer_read_offset = packet_number * kBytesPerPacket * 8;//        +
            //xma::GetPacketFrameOffset(current_input_buffer + packet_number * kBytesPerPacket);
      }
      #endif
    }

    //current_input_buffer += parsed;
    //current_input_size -= parsed;

    // Do the actual packet decoding
    if (packet_->size) {
      auto ret = avcodec_send_packet(context_, packet_);
      if (ret < 0) {
        XELOGE("XmaContext {}: Error sending packet for decoding.", id());
        // TODO bail out
      }
      packet_number++;

      while (ret >= 0) {
        ret = avcodec_receive_frame(context_, decoded_frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
          break;  // TODO ?
        else if (ret < 0) {
          XELOGE("XmaContext {}: Error during decoding.", id());
          break;  // TODO bail out
        }
        //data->input_buffer_read_offset += decoded_frame_->nb_samples / 128;
        auto outfile = fopen(fmt::format("out{}.raw", id()).c_str(), "ab");
        auto data_size = sizeof(float);
        for (int i = 0; i < decoded_frame_->nb_samples; i++)
          for (int ch = 0; ch < context_->channels; ch++)
            fwrite(decoded_frame_->data[ch] + data_size * i, 1, data_size,
                   outfile);
        fclose(outfile);
        // Successfully decoded a frame.
        // Copy to the output buffer.
        size_t written_bytes = 0;

        // TODO !!! Write back buffer to guest. Handle single frames and
        // subframes.

        // Validity checks.
        // assert(decoded_frame_->nb_samples <= kSamplesPerFrame);
        assert(context_->sample_fmt == AV_SAMPLE_FMT_FLTP);

        // Check the returned buffer size.
        /*assert(av_samples_get_buffer_size(NULL, context_->channels,
                                          decoded_frame_->nb_samples,
                                          context_->sample_fmt, 1) ==
               context_->channels * decoded_frame_->nb_samples * sizeof(float));
        */

#if 1
        assert_true(current_frame_.size() == 0);
        size_t raw_count =
            decoded_frame_->nb_samples * kBytesPerSample * num_channels;
        current_frame_.resize(raw_count);
        // Convert the frame.
        ConvertFrame((const uint8_t**)decoded_frame_->data, num_channels,
                     decoded_frame_->nb_samples, current_frame_.data());

        //assert_true(output_remaining_bytes >= kBytesPerFrame * num_channels);
#endif
      }
    }

    if (packet_number * kBytesPerPacket >= current_input_size) {
      if (data->current_buffer == 0) {
        data->input_buffer_0_valid = 0;
        data->input_buffer_read_offset = 0;
        data->current_buffer++;
      } else if (data->current_buffer == 1) {
        data->input_buffer_1_valid = 0;
        data->input_buffer_read_offset = 0;
        data->current_buffer--;
      }
    }
  }


 

  // The game will kick us again with a new output buffer later.
  // It's important that we only invalidate this if we actually wrote to it!!
  if (output_rb.write_offset() == output_rb.read_offset()) {
    data->output_buffer_valid = 0;
  }
}

uint32_t XmaContext::GetFramePacketNumber(uint8_t* block, size_t size,
                                          size_t bit_offset) {
  size *= 8;
  if (bit_offset >= size) {
    // Not good :(
    assert_always();
    return -1;
  }

  size_t byte_offset = bit_offset >> 3;
  size_t packet_number = byte_offset / kBytesPerPacket;

  return (uint32_t)packet_number;
}

int XmaContext::PrepareDecoder(uint8_t* block, size_t size, int sample_rate,
                               int channels) {
  // Sanity check: Packet metadata is always 1 for XMA2/0 for XMA
  assert_true((block[2] & 0x7) == 1 || (block[2] & 0x7) == 0);

  sample_rate = GetSampleRate(sample_rate);

  // Re-initialize the context with new sample rate and channels.
  if (context_->sample_rate != sample_rate || context_->channels != channels) {
    // We have to reopen the codec so it'll realloc whatever data it needs.
    // TODO(DrChat): Find a better way.
    avcodec_close(context_);

    context_->sample_rate = sample_rate;
    context_->channels = channels;
    #if 0
    extra_data_.channel_mask =
        channels == 2 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
    #endif

    if (avcodec_open2(context_, codec_, NULL) < 0) {
      XELOGE("XmaContext: Failed to reopen FFmpeg context");
      return 1;
    }
  }
  return 0;
}

bool XmaContext::ConvertFrame(const uint8_t** samples, int num_channels,
                              int num_samples, uint8_t* output_buffer) {
  // Loop through every sample, convert and drop it into the output array.
  // If more than one channel, we need to interleave the samples from each
  // channel next to each other.
  // TODO: This can definitely be optimized with AVX/SSE intrinsics!
  uint32_t o = 0;
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < num_channels; j++) {
      // Select the appropriate array based on the current channel.
      auto sample_array = reinterpret_cast<const float*>(samples[j]);

      // Raw sample should be within [-1, 1].
      // Clamp it, just in case.
      float raw_sample = xe::saturate(sample_array[i]);

      // Convert the sample and output it in big endian.
      float scaled_sample = raw_sample * ((1 << 15) - 1);
      int sample = static_cast<int>(scaled_sample);
      xe::store_and_swap<uint16_t>(&output_buffer[o++ * 2], sample & 0xFFFF);
    }
  }

  return true;
}

}  // namespace apu
}  // namespace xe
