const express = require('express');
const multer = require('multer');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs').promises;
const ort = require('onnxruntime-node');
const app = express();
const port = process.env.PORT || 3000;

// Constants (Updated for 50MB files)
const MODEL_DIR = path.resolve(__dirname, 'ONNX');
const UPLOAD_DIR = path.resolve(__dirname, 'uploads');
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_FILES = 5; // Max files in upload directory
const FILE_TTL = 3600 * 1000; // 1 hour in milliseconds

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

// Configure multer with 50MB limit
const upload = multer({
  dest: UPLOAD_DIR,
  limits: { 
    fileSize: MAX_FILE_SIZE,
    files: 1 
  },
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.match(/^image\/(jpeg|jpg|png)$/)) {
      return cb(new Error('Only JPEG/PNG images are allowed'), false);
    }
    cb(null, true);
  }
});

// Global state
const models = new Map();
let isInitialized = false;
let initializationAttempted = false;

// Cleanup old files function
async function cleanupUploads() {
  try {
    const files = await fs.readdir(UPLOAD_DIR);
    const now = Date.now();
    
    await Promise.all(files.map(async file => {
      const filePath = path.join(UPLOAD_DIR, file);
      const stats = await fs.stat(filePath);
      if (now - stats.mtimeMs > FILE_TTL) {
        await fs.unlink(filePath).catch(console.error);
      }
    }));

    // If too many files, delete oldest
    if (files.length > MAX_FILES) {
      const sortedFiles = files
        .map(file => ({
          name: file,
          time: fs.statSync(path.join(UPLOAD_DIR, file)).mtimeMs
        }))
        .sort((a, b) => a.time - b.time);

      const toDelete = sortedFiles.slice(0, files.length - MAX_FILES);
      await Promise.all(toDelete.map(file => 
        fs.unlink(path.join(UPLOAD_DIR, file.name)).catch(console.error)
      );
    }
  } catch (err) {
    console.error('Cleanup error:', err);
  }
}

// Initialize application (runs only once)
async function initialize() {
  if (initializationAttempted) return;
  initializationAttempted = true;

  try {
    await fs.mkdir(UPLOAD_DIR, { recursive: true });
    await fs.mkdir(MODEL_DIR, { recursive: true });
    await cleanupUploads();

    // Load models
    console.log(`Loading models from ${MODEL_DIR}`);
    const modelFiles = (await fs.readdir(MODEL_DIR))
      .filter(file => file.endsWith('.onnx'));

    if (modelFiles.length === 0) {
      throw new Error(`No ONNX models found in ${MODEL_DIR}`);
    }

    // Sequential loading for better error handling
    for (const file of modelFiles) {
      const task = file.replace('.onnx', '');
      if (!DOMAIN_TASKS[task]) continue;

      try {
        const modelPath = path.join(MODEL_DIR, file);
        console.log(`Loading ${file}...`);
        models.set(task, await ort.InferenceSession.create(modelPath));
        console.log(`✅ Loaded ${file}`);
      } catch (err) {
        console.error(`❌ Failed to load ${file}:`, err.message);
      }
    }

    if (models.size === 0) {
      throw new Error('No models could be loaded');
    }

    isInitialized = true;
    console.log(`✅ Ready with ${models.size} models loaded`);
  } catch (err) {
    console.error('❌ Initialization failed:', err.message);
    // Don't exit - allow health checks to show status
  }
}

// Image preprocessing (optimized for large files)
async function preprocessImage(imagePath) {
  try {
    const pipeline = sharp(imagePath)
      .resize(224, 224)
      .normalize()
      .removeAlpha()
      .raw();

    const { data, info } = await pipeline.toBuffer({ resolveWithObject: true });

    const floatArray = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
      for (let c = 0; c < 3; c++) {
        const pixel = data[i * 3 + c] / 255.0;
        floatArray[c * 224 * 224 + i] = (pixel - mean[c]) / std[c];
      }
    }

    return floatArray;
  } catch (err) {
    throw new Error(`Image processing failed: ${err.message}`);
  }
}

// API Endpoints
app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  if (!isInitialized) {
    await fs.unlink(req.file.path).catch(() => {});
    return res.status(503).json({ 
      error: 'Service initializing',
      loaded_models: models.size
    });
  }

  try {
    const startTime = Date.now();
    const imageData = await preprocessImage(req.file.path);
    
    // Run predictions
    const results = [];
    for (const [task, domain] of Object.entries(DOMAIN_TASKS)) {
      if (!models.has(task)) continue;

      try {
        const session = models.get(task);
        const input = new ort.Tensor('float32', imageData, [1, 3, 224, 224]);
        const output = await session.run({ input });
        const prediction = output[session.outputNames[0]].data;
        results.push({
          domain,
          confidence: prediction[1], // Assuming binary classification
          model: task
        });
      } catch (err) {
        console.error(`Prediction failed for ${task}:`, err);
      }
    }

    await fs.unlink(req.file.path);

    if (results.length === 0) {
      return res.status(500).json({ error: 'All predictions failed' });
    }

    // Get best result
    const bestResult = results.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );

    res.json({
      status: 'success',
      diagnosis: bestResult.confidence > 0.5 ? 'Malignant' : 'Benign',
      confidence: bestResult.confidence,
      domain: bestResult.domain,
      processing_time_ms: Date.now() - startTime
    });

  } catch (err) {
    await fs.unlink(req.file.path).catch(() => {});
    console.error('Prediction error:', err);
    res.status(500).json({ 
      error: 'Prediction failed',
      details: err.message 
    });
  }
});

// Health endpoint
app.get('/health', (req, res) => {
  res.json({
    status: isInitialized ? 'ready' : 'initializing',
    models_loaded: models.size,
    models_expected: Object.keys(DOMAIN_TASKS).length,
    max_file_size: `${MAX_FILE_SIZE / (1024 * 1024)}MB`
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
initialize().then(() => {
  // Periodic cleanup
  setInterval(cleanupUploads, FILE_TTL / 2).unref();
  
  app.listen(port, () => {
    console.log(`🚀 Server running on port ${port}`);
    console.log(`📂 Accepting files up to ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
  });
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down...');
  process.exit(0);
});