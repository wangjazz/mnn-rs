# mnn-rs VOICEVOX Fork Changelog

此 fork 包含為 VOICEVOX Core MNN Runtime 整合所做的修改。

## 分支：`fix-gru-empty-sequence-lens`

基於 `main` 分支 (commit: `1a275e7` - Updated to mnn 3.2.0)

### 新增功能

#### 1. Module API 支援 (用於包含子圖的模型)

**檔案修改：**
- `mnn-sys/mnn_c/module_c.h` - C API 頭文件
- `mnn-sys/mnn_c/module_c.cpp` - C API 實現
- `src/module.rs` - Rust 封裝

**新增類型：**

```rust
/// Module - 用於包含 Loop/If 等子圖的模型
pub struct Module { ... }

impl Module {
    /// 從檔案載入模型（自動檢測輸入/輸出）
    pub fn load_auto(path: &Path, config: Option<&ModuleConfig>) -> Result<Self>;

    /// 從檔案載入模型（指定輸入/輸出名稱）
    pub fn load(
        path: &Path,
        input_names: &[&str],
        output_names: &[&str],
        config: Option<&ModuleConfig>,
    ) -> Result<Self>;

    /// 執行前向推理
    pub fn forward(&self, inputs: &[&Varp]) -> Result<Vec<Varp>>;

    /// 獲取輸入名稱
    pub fn input_names(&self) -> Vec<String>;

    /// 獲取輸出名稱
    pub fn output_names(&self) -> Vec<String>;
}

/// Varp - Module API 的輸入/輸出張量
pub struct Varp { ... }

impl Varp {
    pub fn new_f32(dims: &[i32]) -> Result<Self>;
    pub fn new_i32(dims: &[i32]) -> Result<Self>;
    pub fn new_i64(dims: &[i32]) -> Result<Self>;
    pub fn new_scalar_i32() -> Result<Self>;  // 標量支援
    pub fn new_scalar_f32() -> Result<Self>;  // 標量支援

    pub fn write_f32(&self, data: &[f32]) -> Result<()>;
    pub fn write_i32(&self, data: &[i32]) -> Result<()>;
    pub fn write_i64(&self, data: &[i64]) -> Result<()>;

    pub fn read_f32(&self, data: &mut [f32]) -> Result<()>;
    pub fn read_i32(&self, data: &mut [i32]) -> Result<()>;

    pub fn element_count(&self) -> usize;
}

/// ModuleConfig - Module 配置
pub struct ModuleConfig {
    pub forward_type: ForwardType,
    pub num_threads: i32,
    pub dynamic: bool,
    pub shape_mutable: bool,
    pub rearrange: bool,
}

impl ModuleConfig {
    pub fn cpu(num_threads: i32) -> Self;
}
```

#### 2. 標量 VARP 支援

修改 `VARP_create` 允許 `dimCount == 0`，用於創建真正的標量張量 (`dims=[]`)。

**用途：** ONNX Loop 模型的 `length` 輸入需要標量而非 `dims=[1]`。

#### 3. GRU 空 sequence_lens 修復

更新 MNN vendor submodule 以包含 GRU 節點處理空 sequence_lens 的修復。

**影響：** 使用 GRU 且 sequence_lens 為空的 ONNX 模型（如 VOICEVOX 的 predict_intonation）現在可以正確轉換。

### Commits

```
049f3a7 feat: add scalar VARP support for ONNX Loop models
f5ee45d feat: enhance Module API with auto-detection and debugging
f1aabc5 feat: add Module API for models with subgraphs (Loop, If, etc.)
572bb62 fix: update MNN submodule to include GRU empty sequence_lens fix
```

### 使用範例

```rust
use mnn::module::{Module, ModuleConfig, Varp};

// 載入模型（自動檢測輸入/輸出）
let config = ModuleConfig::cpu(4);
let module = Module::load_auto("model.mnn", Some(&config))?;

println!("輸入: {:?}", module.input_names());
println!("輸出: {:?}", module.output_names());

// 創建輸入張量
let input1 = Varp::new_scalar_i32()?;  // 標量
input1.write_i32(&[5])?;

let input2 = Varp::new_i32(&[5])?;  // 向量
input2.write_i32(&[0, 1, 2, 3, 4])?;

// 執行推理
let inputs = vec![&input1, &input2];
let outputs = module.forward(&inputs)?;

// 讀取輸出
for output in &outputs {
    let count = output.element_count();
    let mut data = vec![0.0f32; count];
    output.read_f32(&mut data)?;
    println!("Output: {:?}", data);
}
```

### 與上游的差異

此 fork 的修改是 **附加性** 的，不影響現有的 Interpreter API。主要新增：

1. `src/module.rs` - 新模組
2. `mnn-sys/mnn_c/module_c.h` - 新 C API
3. `mnn-sys/mnn_c/module_c.cpp` - 新 C API 實現
4. `mnn-sys/build.rs` - 更新編譯配置
5. `mnn-sys/vendor` - 更新 submodule

### 測試狀態

在 VOICEVOX Core 模型上測試：

| 模型 | 狀態 |
|------|------|
| predict_duration | ✅ 通過 |
| predict_intonation (含 Loop) | ✅ 通過 |
| decode | ✅ 通過 |

### 相關 Issue

- 原始問題：MNN 不支援包含 Loop 子圖的模型通過 Interpreter API 執行
- 解決方案：實現 Module API 封裝
