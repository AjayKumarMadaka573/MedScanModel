const express = require('express');
const multer = require('multer');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs').promises;
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

// Prediction Logic with Confidence-based Logic
async function predictTaskLabel(imagePath) {
  let bestTask = null;
  let bestLabel = null;
  let highestConfidence = -Infinity;  // Confidence range: [0, 1]
  const warnings = [];

  for (const [task, domain] of Object.entries(domainTasks)) {
    try {
      // Simulate confidence generation for the task
      // In a real application, replace this with actual model inference or decision logic
      const confidence = Math.random();  // Placeholder: simulate a random confidence value

      if (confidence > highestConfidence) {
        bestTask = domain;
        bestLabel = confidence > 0.5 ? 'Malignant' : 'Benign'; // Adjust threshold as necessary
        highestConfidence = confidence;
      }

    } catch (err) {
      warnings.push(`Error processing ${task}: ${err.message}`);
    }
  }

  return {
    task: bestTask,
    label: bestLabel,
    confidence: highestConfidence,
    warnings: warnings.length > 0 ? warnings : undefined
  };
}

// API Endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded' });
  }

  try {
    const imagePath = req.file.path;

    try {
      await fs.access(imagePath);
    } catch (err) {
      await fs.unlink(imagePath);
      return res.status(500).json({ error: 'Uploaded image not found' });
    }

    const result = await predictTaskLabel(imagePath);
    await fs.unlink(imagePath);

    if (!result.task) {
      return res.status(400).json({
        error: 'No valid predictions made',
        details: result.warnings
      });
    }

    res.json({
      status: 'success',
      result: {
        diagnosis: result.label,
        confidence: result.confidence,
        domain: result.task,
      },
      metadata: {
        processing_time: new Date().toISOString()
      }
    });
  } catch (err) {
    console.error('Prediction error:', err);
    if (req.file) await fs.unlink(req.file.path).catch(() => {});
    res.status(500).json({
      error: 'Prediction failed',
      details: err.message
    });
  }
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('Unhandled server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start Server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('Shutting down server...');
  process.exit(0);
});
