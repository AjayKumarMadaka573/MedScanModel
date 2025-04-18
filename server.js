const express = require('express');
const multer = require('multer');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs').promises;
const ort = require('onnxruntime-node');
const app = express();
const port = process.env.PORT || 3000;

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// Domain Task Mapping
const domainTasks = {
  "bcn_vs_ham_age_u30": "bcn_age_u30",
  "msk_vs_bcn_age_u30": "msk_age_u30",
  "bcn_vs_ham_loc_head_neck": "bcn_loc_head_neck",
  "bcn_vs_msk_headloc": "bcn_loc_head_neck",
  "ham_vs_msk_loc_head_neck": "ham_loc_head_neck",
  "ham_age_u30vsmsk_age_u30": "ham_age_u30",
  "bcn_vs_ham_loc_palms_soles": "bcn_loc_palms_soles"
};

// Model management
const models = {};
const modelDir = path.resolve(__dirname, 'ONNX');

// Load all ONNX models
async function loadModels() {
  try {
    console.log(`Loading models from: ${modelDir}`);
    
    // Verify ONNX directory exists
    try {
      await fs.access(modelDir);
    } catch {
      throw new Error(`ONNX directory not found at: ${modelDir}`);
    }

    // Load each model
    for (const task of Object.keys(domainTasks)) {
      const modelPath = path.join(modelDir, `${task}.onnx`);
      
      try {
        await fs.access(modelPath);
        console.log(`Loading model: ${task}.onnx`);
        models[task] = await ort.InferenceSession.create(modelPath);
      } catch (err) {
        console.warn(`⚠️ Failed to load model for ${task}: ${err.message}`);
        continue;
      }
    }

    // Verify at least one model loaded
    if (Object.keys(models).length === 0) {
      throw new Error('No ONNX models could be loaded');
    }

    console.log(`Successfully loaded ${Object.keys(models).length} models`);
  } catch (err) {
    console.error('Model loading failed:', err);
    throw err;
  }
}

// Image Preprocessing
async function preprocessImage(imagePath) {
  try {
    const imageBuffer = await sharp(imagePath)
      .resize(224, 224)
      .removeAlpha()
      .raw()
      .toBuffer();

    const floatArray = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
      for (let c = 0; c < 3; c++) {
        const pixel = imageBuffer[i * 3 + c] / 255.0;
        floatArray[c * 224 * 224 + i] = (pixel - mean[c]) / std[c];
      }
    }

    return floatArray;
  } catch (err) {
    throw new Error(`Image preprocessing failed: ${err.message}`);
  }
}

// Prediction Logic
async function predictTaskLabel(imagePath) {
  let bestResult = {
    task: null,
    label: null,
    confidence: -Infinity
  };
  const warnings = [];

  try {
    const imageData = await preprocessImage(imagePath);

    for (const [task, domain] of Object.entries(domainTasks)) {
      if (!models[task]) {
        warnings.push(`Model not loaded for task: ${task}`);
        continue;
      }

      try {
        const model = models[task];
        const inputTensor = new ort.Tensor('float32', imageData, [1, 3, 224, 224]);
        
        // Run inference (note the correct input name)
        const output = await model.run({ input: inputTensor });
        
        // Get the first output tensor
        const outputName = model.outputNames[0];
        const outputData = output[outputName].data;
        
        // Assuming binary classification output [prob_class0, prob_class1]
        const confidence = outputData[1]; // Probability for class 1 (Malignant)
        const label = confidence > 0.5 ? 'Malignant' : 'Benign';

        if (confidence > bestResult.confidence) {
          bestResult = {
            task: domain,
            label,
            confidence
          };
        }
      } catch (err) {
        warnings.push(`Error processing ${task}: ${err.message}`);
      }
    }

    return {
      ...bestResult,
      warnings: warnings.length > 0 ? warnings : undefined
    };

  } catch (err) {
    throw new Error(`Prediction failed: ${err.message}`);
  }
}

// API Endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded' });
  }

  try {
    const result = await predictTaskLabel(req.file.path);
    await fs.unlink(req.file.path);

    if (!result.task) {
      return res.status(400).json({
        error: 'No valid predictions made',
        details: result.warnings || 'No models available'
      });
    }

    res.json({
      status: 'success',
      result: {
        diagnosis: result.label,
        confidence: result.confidence,
        domain: result.task
      },
      metadata: {
        processing_time: new Date().toISOString()
      }
    });
  } catch (err) {
    console.error('Prediction error:', err);
    if (req.file) await fs.unlink(req.file.path).catch(console.error);
    res.status(500).json({
      error: 'Prediction failed',
      details: err.message
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    loaded_models: Object.keys(models).length,
    total_models: Object.keys(domainTasks).length
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server after loading models
loadModels().then(() => {
  app.listen(port, () => {
    console.log(`Server running on port ${port}`);
    console.log(`Loaded ${Object.keys(models).length}/${Object.keys(domainTasks).length} models`);
  });
}).catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  process.exit(0);
});