/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XAM_H_
#define XENIA_KERNEL_XAM_H_

#include "xenia/cpu/export_resolver.h"
#include "xenia/kernel/objects/xkernel_module.h"
#include "xenia/kernel/xam_ordinals.h"

namespace xe {
namespace kernel {

class XamModule : public XKernelModule {
 public:
  XamModule(Emulator* emulator, KernelState* kernel_state);
  virtual ~XamModule();

  static void RegisterExportTable(xe::cpu::ExportResolver* export_resolver);

  struct LoaderData {
    uint32_t launch_data_ptr = 0;
    uint32_t launch_data_size = 0;
    uint32_t launch_flags = 0;
    std::string launch_path;  // Full path to next xex
  };

  const LoaderData& loader_data() const { return loader_data_; }
  LoaderData& loader_data() { return loader_data_; }

 private:
  LoaderData loader_data_;
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XAM_H_
