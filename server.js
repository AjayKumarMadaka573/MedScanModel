// require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs').promises;
const ort = require('onnxruntime-node');
const app = express();
const port = process.env.PORT || 3000;

// Constants
const MODEL_DIR = path.resolve(__dirname, 'ONNX');
const UPLOAD_DIR = path.resolve(__dirname, 'uploads');
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

// Domain Task Mapping
const DOMAIN_TASKS = {
  "bcn_vs_ham_age_u30": "bcn_age_u30",
  "msk_vs_bcn_age_u30": "msk_age_u30",
  "bcn_vs_ham_loc_head_neck": "bcn_loc_head_neck",
  "bcn_vs_msk_headloc": "bcn_loc_head_neck",
  "ham_vs_msk_loc_head_neck": "ham_loc_head_neck",
  "ham_age_u30vsmsk_age_u30": "ham_age_u30",
  "bcn_vs_ham_loc_palms_soles": "bcn_loc_palms_soles"
};

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.match(/^image\/(jpeg|jpg|png)$/)) {
      return cb(new Error('Only image files are allowed!'), false);
    }
    cb(null, true);
  }
});

// Global model cache
const models = new Map();
let isInitialized = false;
let modelsLoading = false;
let modelsLoaded = false;


// Initialize application
async function initialize() {
  if (isInitialized) return;  // Skip initialization if it's already done

  try {
    // Create required directories
    await fs.mkdir(UPLOAD_DIR, { recursive: true });
    await fs.mkdir(MODEL_DIR, { recursive: true });

    // Load models
    await loadModels();
    isInitialized = true;  // Set initialization flag to true
    console.log('✅ Application initialized');
  } catch (err) {
    console.error('❌ Initialization failed:', err);
    process.exit(1);
  }
}

// Load all ONNX models
async function loadModels() {
  // Check if models are already loaded or currently loading
  if (modelsLoaded) {
    console.log('ℹ️ Models already loaded');
    return;
  }
  if (modelsLoading) {
    console.log('⏳ Models are currently being loaded');
    return;
  }

  modelsLoading = true;
  console.log(`🔍 Loading models from: ${MODEL_DIR}`);

  try {
    // Verify model directory exists
    try {
      await fs.access(MODEL_DIR);
    } catch {
      throw new Error(`Model directory not found: ${MODEL_DIR}`);
    }

    // Get list of available models
    const modelFiles = await fs.readdir(MODEL_DIR);
    const onnxModels = modelFiles.filter(file => file.endsWith('.onnx'));

    if (onnxModels.length === 0) {
      throw new Error(`No ONNX models found in ${MODEL_DIR}`);
    }

    // Load models with progress tracking
    const loadingResults = await Promise.allSettled(
      onnxModels.map(async (modelFile) => {
        const task = modelFile.replace('.onnx', '');
        if (!DOMAIN_TASKS[task]) {
          console.warn(`⚠️ No task mapping for model: ${modelFile}`);
          return { task, status: 'skipped' };
        }

        try {
          const modelPath = path.join(MODEL_DIR, modelFile);
          console.log(`⏳ Loading model: ${modelFile}`);
          const session = await ort.InferenceSession.create(modelPath);
          models.set(task, session);
          console.log(`✅ Loaded model: ${modelFile}`);
          return { task, status: 'loaded' };
        } catch (err) {
          console.error(`❌ Failed to load model ${modelFile}:`, err.message);
          return { task, status: 'failed', error: err.message };
        }
      })
    );

    // Analyze loading results
    const loadedCount = loadingResults.filter(
      r => r.value?.status === 'loaded'
    ).length;
    const failedCount = loadingResults.filter(
      r => r.value?.status === 'failed'
    ).length;

    if (loadedCount === 0) {
      throw new Error('No models could be loaded');
    }

    console.log(`\n🎉 Model Loading Summary:`);
    console.log(`   ➔ Success: ${loadedCount}`);
    console.log(`   ➔ Failed: ${failedCount}`);
    console.log(`   ➔ Skipped: ${loadingResults.length - loadedCount - failedCount}\n`);

    modelsLoaded = true;
    return {
      total: onnxModels.length,
      loaded: loadedCount,
      failed: failedCount
    };
  } catch (err) {
    console.error('❌ Model loading failed:', err.message);
    throw err;
  } finally {
    modelsLoading = false;
  }
}




// Image Preprocessing
async function preprocessImage(imagePath) {
  try {
    const imageBuffer = await sharp(imagePath)
      .resize(224, 224)
      .normalize()
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

// Run Inference
async function runInference(task, inputTensor) {
  const session = models.get(task);
  if (!session) {
    throw new Error(`Model not loaded for task: ${task}`);
  }

  try {
    const input = new ort.Tensor('float32', inputTensor, [1, 3, 224, 224]);
    const output = await session.run({ input });
    return output[session.outputNames[0]].data;
  } catch (err) {
    throw new Error(`Inference failed for ${task}: ${err.message}`);
  }
}

// Prediction Logic
async function predictTaskLabel(imagePath) {
  const results = [];
  const warnings = [];

  try {
    const imageData = await preprocessImage(imagePath);

    for (const [task, domain] of Object.entries(DOMAIN_TASKS)) {
      if (!models.has(task)) {
        warnings.push(`Model not available for task: ${task}`);
        continue;
      }

      try {
        const output = await runInference(task, imageData);
        const confidence = output[1]; // Assuming binary classification
        results.push({
          task,
          domain,
          confidence,
          label: confidence > 0.5 ? 'Malignant' : 'Benign'
        });
      } catch (err) {
        warnings.push(`Error processing ${task}: ${err.message}`);
      }
    }

    if (results.length === 0) {
      throw new Error('No successful predictions');
    }

    // Find result with highest confidence
    const bestResult = results.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );

    return {
      ...bestResult,
      warnings: warnings.length > 0 ? warnings : undefined
    };
  } catch (err) {
    throw new Error(`Prediction failed: ${err.message}`);
  }
}

// API Endpoints
app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ 
      status: 'error',
      error: 'No image file uploaded' 
    });
  }

  if (!isInitialized) {
    return res.status(503).json({
      status: 'error',
      error: 'Service initializing'
    });
  }

  try {
    const result = await predictTaskLabel(req.file.path);
    
    // Clean up uploaded file
    await fs.unlink(req.file.path).catch(console.error);

    res.json({
      status: 'success',
      result: {
        diagnosis: result.label,
        confidence: result.confidence,
        domain: result.domain
      },
      metadata: {
        model: result.task,
        processing_time: new Date().toISOString()
      },
      warnings: result.warnings
    });
  } catch (err) {
    console.error('Prediction error:', err);
    await fs.unlink(req.file.path).catch(console.error);
    res.status(500).json({
      status: 'error',
      error: 'Prediction failed',
      details: err.message
    });
  }
});

// Health Check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    models_loaded: models.size,
    models_expected: Object.keys(DOMAIN_TASKS).length,
    initialized: isInitialized
  });
});

// Error Handling Middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ 
    status: 'error',
    error: 'Internal server error' 
  });
});

// Start Server
initialize().then(() => {
  app.listen(port, () => {
    console.log(`🚀 Server running on port ${port}`);
    console.log(`📁 Model directory: ${MODEL_DIR}`);
    console.log(`📁 Upload directory: ${UPLOAD_DIR}`);
  });
});

// Graceful Shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down server...');
  try {
    // Clean up resources
    await fs.rm(UPLOAD_DIR, { recursive: true, force: true });
    console.log('✅ Cleanup complete');
    process.exit(0);
  } catch (err) {
    console.error('Shutdown error:', err);
    process.exit(1);
  }
});
