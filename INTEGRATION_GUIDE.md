# ISRO Thermal SR - Integration Guide

## 🚀 Quick Start

Your Next.js app is now integrated with your trained thermal super-resolution model!

### 1. Start the Python API Server
```bash
cd thermal_sr
python api_server.py
```
This starts the model server on `http://localhost:8000`

### 2. Start the Next.js App
```bash
npm run dev
```
This starts your web app on `http://localhost:3000`

### 3. Use the Application
1. Go to `http://localhost:3000/fuse-sr`
2. Upload a thermal image (low resolution)
3. Upload an optical image (high resolution)
4. Click "Run Super-Resolution"
5. View the enhanced thermal image result

## 📁 Files Added/Modified

### New API Files:
- `thermal_sr/api_server.py` - FastAPI server serving your trained model
- `thermal_sr/thermal_sr_model.pkl` - Your trained model (305KB)
- `app/api/thermal-sr/route.js` - Next.js API route

### Modified Files:
- `app/fuse-sr/page.js` - Updated to use real model with file uploads

## 🔧 How It Works

1. **Frontend**: User uploads thermal + optical images in Next.js
2. **Next.js API**: Forwards files to Python API server
3. **Python API**: Loads your trained model and processes images
4. **Model**: Performs 2× super-resolution using optical guidance
5. **Response**: Returns enhanced thermal image as base64

## 🎯 Model Features

- **Input**: 128×128 thermal + 256×256 optical RGB
- **Output**: 256×256 enhanced thermal image
- **Architecture**: Alignment-Fusion CNN with attention gates
- **Processing Time**: ~1-2 seconds on GPU

## 🛠️ Troubleshooting

### API Server Won't Start
```bash
cd thermal_sr
pip install fastapi uvicorn python-multipart torch pillow opencv-python
python api_server.py
```

### Model Not Found
Make sure `thermal_sr_model.pkl` exists in the `thermal_sr` directory.

### CORS Issues
The API server is configured to allow requests from `localhost:3000`.

## 📊 Testing

Test the API directly:
```bash
cd thermal_sr
python test_api.py
```

## 🔄 Using Different Models

To use a different model:
1. Train your model and save as pickle
2. Update the model path in `api_server.py`
3. Restart the API server

Your thermal super-resolution system is now fully integrated and ready to use!