/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/cpu/processor.h>

#include <xenia/cpu/jit.h>
#include <xenia/cpu/ppc/disasm.h>


using namespace xe;
using namespace xe::cpu;
using namespace xe::kernel;


namespace {
  void InitializeIfNeeded();
  void CleanupOnShutdown();

  void InitializeIfNeeded() {
    static bool has_initialized = false;
    if (has_initialized) {
      return;
    }
    has_initialized = true;

    ppc::RegisterDisasmCategoryALU();
    ppc::RegisterDisasmCategoryControl();
    ppc::RegisterDisasmCategoryFPU();
    ppc::RegisterDisasmCategoryMemory();

    atexit(CleanupOnShutdown);
  }

  void CleanupOnShutdown() {
  }
}


Processor::Processor(xe_memory_ref memory, shared_ptr<Backend> backend) :
    fn_table_(NULL), jit_(NULL) {
  memory_ = xe_memory_retain(memory);
  backend_ = backend;

  InitializeIfNeeded();
}

Processor::~Processor() {
  // Cleanup all modules.
  for (std::vector<ExecModule*>::iterator it = modules_.begin();
       it != modules_.end(); ++it) {
    ExecModule* exec_module = *it;
    if (jit_) {
      jit_->UninitModule(exec_module);
    }
    delete exec_module;
  }
  modules_.clear();

  delete jit_;
  delete fn_table_;

  export_resolver_.reset();
  backend_.reset();
  xe_memory_release(memory_);
}

xe_memory_ref Processor::memory() {
  return xe_memory_retain(memory_);
}

shared_ptr<ExportResolver> Processor::export_resolver() {
  return export_resolver_;
}

void Processor::set_export_resolver(
    shared_ptr<ExportResolver> export_resolver) {
  export_resolver_ = export_resolver;
}

int Processor::Setup() {
  XEASSERTNULL(jit_);

  fn_table_ = new FunctionTable();

  jit_ = backend_->CreateJIT(memory_, fn_table_);
  if (jit_->Setup()) {
    XELOGE("Unable to create JIT");
    return 1;
  }

  return 0;
}

int Processor::LoadRawBinary(const xechar_t* path, uint32_t start_address) {
  ExecModule* exec_module = NULL;
  const xechar_t* name = xestrrchr(path, '/') + 1;

  // TODO(benvanik): map file from filesystem API, not via platform API.
  xe_mmap_ref mmap = xe_mmap_open(kXEFileModeRead, path, 0, 0);
  if (!mmap) {
    return NULL;
  }
  void* addr = xe_mmap_get_addr(mmap);
  size_t length = xe_mmap_get_length(mmap);

  int result_code = 1;

  // Place the data into memory at the desired address.
  XEEXPECTZERO(xe_copy_memory(xe_memory_addr(memory_, start_address),
                              xe_memory_get_length(memory_),
                              addr, length));

  char name_a[XE_MAX_PATH];
  XEEXPECTTRUE(xestrnarrow(name_a, XECOUNT(name_a), name));
  char path_a[XE_MAX_PATH];
  XEEXPECTTRUE(xestrnarrow(path_a, XECOUNT(path_a), path));

  // Prepare the module.
  // This will analyze it, generate code (if needed), and adds methods to
  // the function table.
  exec_module = new ExecModule(
      memory_, export_resolver_, fn_table_, name_a, path_a);
  XEEXPECTZERO(exec_module->PrepareRawBinary(
      start_address, start_address + (uint32_t)length));

  // Initialize the module and prepare it for execution.
  XEEXPECTZERO(jit_->InitModule(exec_module));

  modules_.push_back(exec_module);

  result_code = 0;
XECLEANUP:
  if (result_code) {
    delete exec_module;
  }
  xe_mmap_release(mmap);
  return result_code;
}

int Processor::LoadXexModule(const char* name, const char* path,
                             xe_xex2_ref xex) {
  int result_code = 1;

  // Prepare the module.
  // This will analyze it, generate code (if needed), and adds methods to
  // the function table.
  ExecModule* exec_module = new ExecModule(
      memory_, export_resolver_, fn_table_, name, path);
  XEEXPECTZERO(exec_module->PrepareXexModule(xex));

  // Initialize the module and prepare it for execution.
  XEEXPECTZERO(jit_->InitModule(exec_module));

  modules_.push_back(exec_module);

  result_code = 0;
XECLEANUP:
  if (result_code) {
    delete exec_module;
  }
  return result_code;
}

uint32_t Processor::CreateCallback(void (*callback)(void* data), void* data) {
  // TODO(benvanik): implement callback creation.
  return 0;
}

ThreadState* Processor::AllocThread(uint32_t stack_size,
                                    uint32_t thread_state_address) {
  ThreadState* thread_state = new ThreadState(
      this, stack_size, thread_state_address);
  return thread_state;
}

void Processor::DeallocThread(ThreadState* thread_state) {
  delete thread_state;
}

FunctionPointer Processor::GenerateFunction(uint32_t address) {
  // Search all modules for the function symbol.
  // Each module will see if the address is within its code range and if the
  // symbol is not found (likely) it will do analysis on it.
  // TODO(benvanik): make this more efficient. Could use a binary search or
  //     something more clever.
  sdb::FunctionSymbol* fn_symbol = NULL;
  for (std::vector<ExecModule*>::iterator it = modules_.begin();
      it != modules_.end(); ++it) {
    fn_symbol = (*it)->FindFunctionSymbol(address);
    if (fn_symbol) {
      break;
    }
  }
  if (!fn_symbol) {
    return NULL;
  }

  // JIT the function.
  FunctionPointer f = jit_->GenerateFunction(fn_symbol);
  if (!f) {
    return NULL;
  }

  return f;
}

int Processor::Execute(ThreadState* thread_state, uint32_t address) {
  xe_ppc_state_t* ppc_state = thread_state->ppc_state();

  // This could be set to anything to give us a unique identifier to track
  // re-entrancy/etc.
  uint32_t lr = 0xBEBEBEBE;

  // Setup registers.
  ppc_state->lr = lr;

  // Find the function to execute.
  FunctionPointer f = fn_table_->GetFunction(address);

  // JIT, if needed.
  if (!f) {
    f = this->GenerateFunction(address);
  }

  // If JIT failed, die.
  if (!f) {
    XELOGCPU("Execute(%.8X): failed to find function", address);
    return 1;
  }

  // Execute the function pointer.
  // Messes with the stack in such a way as to cause Xcode to behave oddly.
  f(ppc_state, lr);

  return 0;
}

uint64_t Processor::Execute(ThreadState* thread_state, uint32_t address,
                       uint64_t arg0) {
  xe_ppc_state_t* ppc_state = thread_state->ppc_state();
  ppc_state->r[3] = arg0;
  if (Execute(thread_state, address)) {
    return 0xDEADBABE;
  }
  return ppc_state->r[3];
}
