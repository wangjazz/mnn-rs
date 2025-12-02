//! MNN Express Module API for models with subgraphs (Loop, If, etc.)
//!
//! This module provides a high-level API for running MNN models that contain
//! dynamic control flow operations like Loop and If (subgraphs).
//!
//! # Example
//! ```rust,ignore
//! use mnn::module::{Module, ModuleConfig, Varp};
//!
//! // Load module with input/output names
//! let config = ModuleConfig::default();
//! let module = Module::load(
//!     "model.mnn",
//!     &["input1", "input2"],
//!     &["output"],
//!     Some(&config)
//! ).unwrap();
//!
//! // Create input tensors
//! let input1 = Varp::new_f32(&[1, 10]).unwrap();
//! let data = vec![0.0f32; 10];
//! input1.write_f32(&data).unwrap();
//!
//! // Run forward pass
//! let outputs = module.forward(&[&input1]).unwrap();
//!
//! // Read output
//! let mut output_data = vec![0.0f32; outputs[0].element_count()];
//! outputs[0].read_f32(&mut output_data).unwrap();
//! ```

use crate::error::{ensure, error};
use crate::{ErrorKind, Result};
use error_stack::ResultExt;
use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;

/// Module configuration
#[derive(Debug, Clone)]
pub struct ModuleConfig {
    /// Load module as dynamic (default: false = static)
    pub dynamic: bool,
    /// For static mode, if the shape is mutable (default: true)
    pub shape_mutable: bool,
    /// Pre-rearrange weights or not (default: false)
    pub rearrange: bool,
    /// Backend type
    pub backend_type: mnn_sys::MNNForwardType,
    /// Number of threads (for CPU backend)
    pub num_threads: i32,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            dynamic: false,
            shape_mutable: true,
            rearrange: false,
            backend_type: mnn_sys::MNNForwardType::MNN_FORWARD_CPU,
            num_threads: 4,
        }
    }
}

impl ModuleConfig {
    /// Create a new module configuration for CPU backend
    pub fn cpu(num_threads: i32) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }

    /// Create a new module configuration for Metal backend (macOS/iOS)
    #[cfg(feature = "metal")]
    pub fn metal() -> Self {
        Self {
            backend_type: mnn_sys::MNNForwardType::MNN_FORWARD_METAL,
            ..Default::default()
        }
    }

    /// Create a new module configuration for OpenCL backend
    #[cfg(feature = "opencl")]
    pub fn opencl() -> Self {
        Self {
            backend_type: mnn_sys::MNNForwardType::MNN_FORWARD_OPENCL,
            ..Default::default()
        }
    }
}

/// MNN Express Variable (VARP) - represents an input or output tensor
pub struct Varp {
    inner: NonNull<mnn_sys::VARP>,
    owned: bool,
}

impl Varp {
    /// Create a new float32 VARP with given shape
    pub fn new_f32(dims: &[i32]) -> Result<Self> {
        Self::new(dims, 0) // 0 = float32
    }

    /// Create a new int32 VARP with given shape
    pub fn new_i32(dims: &[i32]) -> Result<Self> {
        Self::new(dims, 1) // 1 = int32
    }

    /// Create a new int64 VARP with given shape
    pub fn new_i64(dims: &[i32]) -> Result<Self> {
        Self::new(dims, 2) // 2 = int64
    }

    /// Create a new scalar int32 VARP (dims=[])
    pub fn new_scalar_i32() -> Result<Self> {
        Self::new(&[], 1) // 1 = int32, empty dims = scalar
    }

    /// Create a new scalar float32 VARP (dims=[])
    pub fn new_scalar_f32() -> Result<Self> {
        Self::new(&[], 0) // 0 = float32, empty dims = scalar
    }

    fn new(dims: &[i32], data_type: i32) -> Result<Self> {
        let ptr = unsafe { mnn_sys::VARP_create(dims.as_ptr(), dims.len(), data_type) };

        NonNull::new(ptr)
            .map(|inner| Self { inner, owned: true })
            .ok_or_else(|| error!(ErrorKind::TensorError).attach_printable("Failed to create VARP"))
    }

    fn from_raw(ptr: *mut mnn_sys::VARP) -> Option<Self> {
        NonNull::new(ptr).map(|inner| Self { inner, owned: true })
    }

    /// Write float32 data to the VARP
    pub fn write_f32(&self, data: &[f32]) -> Result<()> {
        let ret = unsafe {
            mnn_sys::VARP_writeFloat(self.inner.as_ptr(), data.as_ptr(), data.len())
        };

        ensure!(
            ret == 0,
            ErrorKind::TensorError;
            "Failed to write float data to VARP"
        );
        Ok(())
    }

    /// Write int32 data to the VARP
    pub fn write_i32(&self, data: &[i32]) -> Result<()> {
        let ret = unsafe {
            mnn_sys::VARP_writeInt32(self.inner.as_ptr(), data.as_ptr(), data.len())
        };

        ensure!(
            ret == 0,
            ErrorKind::TensorError;
            "Failed to write int32 data to VARP"
        );
        Ok(())
    }

    /// Write int64 data to the VARP
    pub fn write_i64(&self, data: &[i64]) -> Result<()> {
        let ret = unsafe {
            mnn_sys::VARP_writeInt64(self.inner.as_ptr(), data.as_ptr(), data.len())
        };

        ensure!(
            ret == 0,
            ErrorKind::TensorError;
            "Failed to write int64 data to VARP"
        );
        Ok(())
    }

    /// Read float32 data from the VARP
    pub fn read_f32(&self, data: &mut [f32]) -> Result<()> {
        let ret = unsafe {
            mnn_sys::VARP_readFloat(self.inner.as_ptr(), data.as_mut_ptr(), data.len())
        };

        ensure!(
            ret == 0,
            ErrorKind::TensorError;
            "Failed to read float data from VARP"
        );
        Ok(())
    }

    /// Read int32 data from the VARP
    pub fn read_i32(&self, data: &mut [i32]) -> Result<()> {
        let ret = unsafe {
            mnn_sys::VARP_readInt32(self.inner.as_ptr(), data.as_mut_ptr(), data.len())
        };

        ensure!(
            ret == 0,
            ErrorKind::TensorError;
            "Failed to read int32 data from VARP"
        );
        Ok(())
    }

    /// Get the number of elements in the VARP
    pub fn element_count(&self) -> usize {
        unsafe { mnn_sys::VARP_getElementCount(self.inner.as_ptr()) }
    }

    /// Get the raw pointer (for passing to Module_forward)
    fn as_ptr(&self) -> *mut mnn_sys::VARP {
        self.inner.as_ptr()
    }
}

impl Drop for Varp {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                mnn_sys::VARP_destroy(self.inner.as_ptr());
            }
        }
    }
}

// VARP is not thread-safe by default
// unsafe impl Send for Varp {}

/// MNN Express Module for models with subgraphs
pub struct Module {
    inner: NonNull<mnn_sys::Module>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl Module {
    /// Load a module from file with auto-detection of input/output names
    ///
    /// When input_names and output_names are empty, MNN will automatically
    /// detect them from the model file.
    pub fn load_auto<P: AsRef<Path>>(path: P, config: Option<&ModuleConfig>) -> Result<Self> {
        Self::load(path, &[], &[], config)
    }

    /// Load a module from file
    pub fn load<P: AsRef<Path>>(
        path: P,
        input_names: &[&str],
        output_names: &[&str],
        config: Option<&ModuleConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();
        ensure!(path.exists(), ErrorKind::IOError; path.to_string_lossy().to_string(), "File not found");

        let path_str = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = CString::new(path_str).change_context(ErrorKind::AsciiError)?;

        let input_cstrings: Vec<CString> = input_names
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<Vec<_>, _>>()
            .change_context(ErrorKind::AsciiError)?;

        let output_cstrings: Vec<CString> = output_names
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<Vec<_>, _>>()
            .change_context(ErrorKind::AsciiError)?;

        let mut input_ptrs: Vec<*const i8> = input_cstrings.iter().map(|s| s.as_ptr()).collect();

        let mut output_ptrs: Vec<*const i8> = output_cstrings.iter().map(|s| s.as_ptr()).collect();

        let c_config = config.map(|c| mnn_sys::ModuleConfig {
            dynamic: if c.dynamic { 1 } else { 0 },
            shapeMutable: if c.shape_mutable { 1 } else { 0 },
            rearrange: if c.rearrange { 1 } else { 0 },
            backendType: c.backend_type,
            numThreads: c.num_threads,
        });

        // Allow null pointers for empty name lists
        let input_ptr = if input_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            input_ptrs.as_mut_ptr()
        };
        let output_ptr = if output_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            output_ptrs.as_mut_ptr()
        };

        let ptr = unsafe {
            mnn_sys::Module_loadFromFile(
                c_path.as_ptr(),
                input_ptr,
                input_ptrs.len(),
                output_ptr,
                output_ptrs.len(),
                c_config
                    .as_ref()
                    .map(|c| c as *const _)
                    .unwrap_or(std::ptr::null()),
            )
        };

        ensure!(
            !ptr.is_null(),
            ErrorKind::InterpreterError;
            format!("Failed to load module from file: {}", path.display())
        );

        // Get the actual input/output names from the module info
        let module_info = unsafe { mnn_sys::Module_getInfo(ptr) };
        let (actual_input_names, actual_output_names) = if !module_info.is_null() {
            let info = unsafe { &*module_info };
            let inputs = Self::extract_names(info.inputNames, info.inputCount);
            let outputs = Self::extract_names(info.outputNames, info.outputCount);
            (inputs, outputs)
        } else {
            (
                input_names.iter().map(|s| s.to_string()).collect(),
                output_names.iter().map(|s| s.to_string()).collect(),
            )
        };

        Ok(Self {
            inner: NonNull::new(ptr).unwrap(),
            input_names: actual_input_names,
            output_names: actual_output_names,
        })
    }

    /// Extract string vector from C string array
    fn extract_names(names: *const *const i8, count: usize) -> Vec<String> {
        if names.is_null() || count == 0 {
            return Vec::new();
        }
        (0..count)
            .filter_map(|i| {
                let ptr = unsafe { *names.add(i) };
                if ptr.is_null() {
                    None
                } else {
                    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
                    cstr.to_str().ok().map(|s| s.to_string())
                }
            })
            .collect()
    }

    /// Load a module from buffer
    pub fn load_from_buffer(
        buffer: &[u8],
        input_names: &[&str],
        output_names: &[&str],
        config: Option<&ModuleConfig>,
    ) -> Result<Self> {
        let input_cstrings: Vec<CString> = input_names
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<Vec<_>, _>>()
            .change_context(ErrorKind::AsciiError)?;

        let output_cstrings: Vec<CString> = output_names
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<Vec<_>, _>>()
            .change_context(ErrorKind::AsciiError)?;

        let mut input_ptrs: Vec<*const i8> = input_cstrings.iter().map(|s| s.as_ptr()).collect();

        let mut output_ptrs: Vec<*const i8> = output_cstrings.iter().map(|s| s.as_ptr()).collect();

        let c_config = config.map(|c| mnn_sys::ModuleConfig {
            dynamic: if c.dynamic { 1 } else { 0 },
            shapeMutable: if c.shape_mutable { 1 } else { 0 },
            rearrange: if c.rearrange { 1 } else { 0 },
            backendType: c.backend_type,
            numThreads: c.num_threads,
        });

        let ptr = unsafe {
            mnn_sys::Module_loadFromBuffer(
                buffer.as_ptr() as *const _,
                buffer.len(),
                input_ptrs.as_mut_ptr(),
                input_ptrs.len(),
                output_ptrs.as_mut_ptr(),
                output_ptrs.len(),
                c_config
                    .as_ref()
                    .map(|c| c as *const _)
                    .unwrap_or(std::ptr::null()),
            )
        };

        ensure!(
            !ptr.is_null(),
            ErrorKind::InterpreterError;
            "Failed to load module from buffer"
        );

        Ok(Self {
            inner: NonNull::new(ptr).unwrap(),
            input_names: input_names.iter().map(|s| s.to_string()).collect(),
            output_names: output_names.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Get input tensor names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output tensor names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run forward pass with the given inputs
    pub fn forward(&self, inputs: &[&Varp]) -> Result<Vec<Varp>> {
        let input_ptrs: Vec<*mut mnn_sys::VARP> = inputs.iter().map(|v| v.as_ptr()).collect();

        let output_count = self.output_names.len();
        let mut output_ptrs: Vec<*mut mnn_sys::VARP> = vec![std::ptr::null_mut(); output_count];

        let ret = unsafe {
            mnn_sys::Module_forward(
                self.inner.as_ptr(),
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_mut_ptr(),
                output_count,
            )
        };

        ensure!(
            ret == 0,
            ErrorKind::InterpreterError;
            format!("Module forward failed with error code: {}", ret)
        );

        let outputs: Vec<Varp> = output_ptrs.into_iter().filter_map(Varp::from_raw).collect();

        ensure!(
            outputs.len() == output_count,
            ErrorKind::InterpreterError;
            format!("Expected {} outputs, got {}", output_count, outputs.len())
        );

        Ok(outputs)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            mnn_sys::Module_destroy(self.inner.as_ptr());
        }
    }
}

// Module is not thread-safe by default
// unsafe impl Send for Module {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varp_create() {
        let varp = Varp::new_f32(&[1, 3, 224, 224]).unwrap();
        assert!(varp.element_count() > 0);
    }
}
