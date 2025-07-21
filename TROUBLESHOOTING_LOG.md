# CatVTON-Flux: Troubleshooting & Fixes Log

This document tracks all issues encountered, their root causes, and the solutions implemented during the setup and optimization of the CatVTON-Flux virtual try-on system.

## Issue #1: Gradio Version Compatibility Error

### **Error Description:**
```
TypeError: argument of type 'bool' is not iterable
```
- Error occurred in `gradio_client/utils.py` line 897: `if "const" in schema:`
- Caused app crashes when accessing the web interface

### **Root Cause:**
Version incompatibility between `gradio` and its ecosystem dependencies (`fastapi`, `uvicorn`, `starlette`, `pydantic`). The error occurs when gradio tries to generate API documentation and expects a dictionary but receives a boolean value.

### **Solution Implemented:**
1. **Downgraded Gradio ecosystem to compatible versions:**
   ```bash
   pip install gradio==4.21.0 gradio_client==0.12.0
   pip install fastapi==0.104.1 uvicorn==0.24.0 starlette==0.27.0 pydantic==2.5.0
   ```

2. **Added `share=True` to `demo.launch()`** to create public URLs and avoid localhost issues.

### **Result:** ✅ Fixed - App now launches successfully without compatibility errors.

---

## Issue #2: CUDA Out of Memory Error

### **Error Description:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 72.00 MiB. 
GPU 0 has a total capacity of 44.40 GiB of which 35.00 MiB is free. 
Process 5757 has 31.86 GiB memory in use.
```

### **Root Cause:**
- Previous Python process (PID 5757) was consuming 31.86 GiB of GPU memory
- GPU had only 35 MiB free out of 44.40 GiB total capacity
- Model loading requires significant GPU memory for FLUX transformer

### **Solution Implemented:**
1. **Killed memory-consuming process:**
   ```bash
   kill -9 5757
   nvidia-smi  # Verified 0MiB GPU usage
   ```

2. **Optimized model loading in `app.py`:**
   ```python
   transformer = FluxTransformer2DModel.from_pretrained(
       "xiaozaa/catvton-flux-alpha", 
       torch_dtype=dtype,
       low_cpu_mem_usage=True  # Added this
   )
   pipe = FluxFillPipeline.from_pretrained(
       "black-forest-labs/FLUX.1-dev",
       transformer=transformer,
       torch_dtype=dtype,
       low_cpu_mem_usage=True  # Added this
   )
   pipe = pipe.to(device)  # Moved .to() to separate line
   ```

### **Result:** ✅ Fixed - Model loads successfully with efficient memory usage.

---

## Current Working Configuration

### **Dependencies (Working Versions):**
```
gradio==4.21.0
gradio_client==0.12.0
fastapi==0.104.1
uvicorn==0.24.0
starlette==0.27.0
pydantic==2.5.0
torch==2.5.1
transformers==4.43.3
diffusers (latest from git)
```

### **Hardware Requirements:**
- GPU: NVIDIA L40S (44.40 GiB)
- Memory needed: ~25-30 GiB for full model loading
- CUDA Version: 12.8

### **Working URLs:**
- Local: `http://127.0.0.1:7860`
- Public: `https://a0f27d982cc27e68f2.gradio.live` (expires in 72 hours)

---

## Preventive Measures

### **To Avoid GPU Memory Issues:**
1. Always check GPU usage before launching: `nvidia-smi`
2. Kill stale processes if needed: `kill -9 <PID>`
3. Use `low_cpu_mem_usage=True` in model loading
4. Monitor memory during inference

### **To Avoid Dependency Issues:**
1. Pin all dependency versions in `requirements.txt`
2. Use conda/pip environments for isolation
3. Test compatibility before major upgrades
4. Keep backup of working `requirements.txt`

---

## Model Weights Location

### **Downloaded Weights:**
- **Base FLUX.1-dev**: `~/.cache/huggingface/hub/` (~23 GB)
- **Fine-tuned CatVTON**: `/home/sheldon/cvton/catvton-flux-beta/` (~23 GB)
- **Total disk usage**: ~46 GB

### **To Clean Up (if needed):**
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface/hub/

# Remove local fine-tuned model
rm -rf /home/sheldon/cvton/catvton-flux-beta/
```

---

## Next Steps (Roadmap)

1. ✅ **Phase 1**: Basic app functioning (COMPLETED)
2. 🔄 **Phase 2**: Test full inference pipeline with sample images
3. 📋 **Phase 3**: Implement lower-body support research
4. 🚀 **Phase 4**: Production API development

---

## Contact & Support

For issues or questions:
- Check this troubleshooting log first
- Review `PROJECT_ROADMAP.md` for feature status
- Monitor GPU memory with `nvidia-smi`
- Test with stable dependency versions listed above
