#include "Minimal.h"

#include <ATen/native/Resize.h>
#include <third_party/flagos/include/flagos.h>

namespace at::native::flagos {

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  // Convert SymInt to Int and use native implementation
  auto int_size = C10_AS_INTARRAYREF_SLOW(size);
  auto int_stride = C10_AS_INTARRAYREF_SLOW(stride);
  std::optional<int64_t> int_offset = storage_offset.has_value()
      ? std::optional<int64_t>(storage_offset->expect_int())
      : std::nullopt;
  return at::native::as_strided_tensorimpl(self, int_size, int_stride, int_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  // Get contiguous tensors for copy
  at::Tensor self_contig = self.contiguous();
  at::Tensor dst_contig = dst.is_contiguous() ? dst : dst.contiguous();

  size_t nbytes = self_contig.numel() * self_contig.element_size();

  if (self.is_privateuseone() && dst.is_privateuseone()) {
    // Device to device copy
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
  } else if (self.is_cpu() && dst.is_privateuseone()) {
    // Host to device copy
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyHostToDevice);
  } else if (self.is_privateuseone() && dst.is_cpu()) {
    // Device to host copy
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToHost);
  } else {
    TORCH_CHECK(false, "Unsupported device combination for copy");
  }

  // If dst was not contiguous, copy back
  if (!dst.is_contiguous()) {
    dst.copy_(dst_contig);
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::flagos::_copy_from(self, dst, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  // Copy single element from device to host
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense expects a tensor with 1 element");

  at::Tensor cpu_tensor = at::empty({1}, self.options().device(at::kCPU));
  foMemcpy(cpu_tensor.data_ptr(), self.data_ptr(), self.element_size(), foMemcpyDeviceToHost);

  return cpu_tensor.item();
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  // Set storage and then manually set the storage offset, size, and stride
  result.set_(storage);
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return result;
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // For now, delegate all operations to CPU fallback
  // FlagGems operators will be registered separately in Python
  at::native::cpu_fallback(op, stack);
}

} // namespace at::native::flagos
