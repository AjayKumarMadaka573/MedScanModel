const express = require('express');
const multer = require('multer');
const path = require('path');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;
const NpyJS = require('npyjs'); // For handling .npy files

const app = express();
const port = process.env.PORT || 3000;

const upload = multer({ dest: 'uploads/' });

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
}

// Run ONNX Model Inference Using TensorFlow.js
async function runInference(modelPath, inputTensor) {
  const model = await tf.loadGraphModel(`file://${modelPath}`);
  const inputTensorObj = tf.tensor(inputTensor, [1, 3, 224, 224]);
  const output = model.predict(inputTensorObj);
  return output.dataSync();
}

// Softmax Function for Probability
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
}

// A-distance computation
function averageAdistance(imageFeat, domainFeatures) {
  const norm = (v) => Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
  const normalize = (arr) => {
    const n = norm(arr);
    return arr.map(x => x / n);
  };

  const imageNorm = normalize(imageFeat);
  const domainNorm = domainFeatures.map(feat => normalize(feat));

  const distances = domainNorm.map(domainVec => {
    return Math.sqrt(domainVec.reduce((sum, val, i) => sum + Math.pow(val - imageNorm[i], 2), 0));
  });

  return distances.reduce((a, b) => a + b, 0) / distances.length;
}

// Load Numpy (.npy) Files Using npyjs
async function loadNpyFile(filePath) {
  return new Promise((resolve, reject) => {
    const npy = new NpyJS();
    npy.load(filePath, (err, data) => {
      if (err) {
        return reject(err);
      }
      resolve(data);
    });
  });
}

// Predict Task Label
async function predictTaskLabel(imagePath, modelDir, featDir) {
  let minDistance = Infinity;
  let bestTask = null;
  let bestLabel = null;
  let bestConfidence = 0;

  for (const [task, domain] of Object.entries(domainTasks)) {
    try {
      console.log(`🔍 Evaluating task: ${task}`);

      const modelPath = path.join(modelDir, `${task}.onnx`);
      const featuresPath = path.join(featDir, `${task}_features.npy`);

      try {
        await fs.access(modelPath);
        await fs.access(featuresPath);
      } catch {
        console.warn(`⚠️ Missing model or features for ${task}`);
        continue;
      }

      const imageFeat = await preprocessImage(imagePath);
      const output = await runInference(modelPath, imageFeat);

      const outputTensor = Array.from(output);
      const probabilities = softmax(outputTensor);
      const predictedLabel = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedLabel];

      const domainFeatures = await loadNpyFile(featuresPath);

      if (!Array.isArray(domainFeatures) || !Array.isArray(domainFeatures[0])) {
        throw new Error(`Invalid features array for ${task}`);
      }

      const adist = averageAdistance(imageFeat, domainFeatures);
      console.log(`   → Avg A-distance: ${adist.toFixed(4)}, Confidence: ${confidence.toFixed(4)}`);

      if (adist < minDistance) {
        minDistance = adist;
        bestTask = domain;
        bestLabel = predictedLabel;
        bestConfidence = confidence;
      }

    } catch (err) {
      console.error(`⚠️ Error processing ${task}:`, err.message);
      continue;
    }
  }

  return {
    task: bestTask,
    distance: minDistance,
    label: bestLabel,
    confidence: bestConfidence
  };
}

// API Endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    const imagePath = req.file.path;
    const modelDir = path.resolve(__dirname, 'ONNX');
    const featDir = path.resolve(__dirname, 'domain_features_dann');

    const result = await predictTaskLabel(imagePath, modelDir, featDir);
    await fs.unlink(imagePath);
    res.json(result);
  } catch (err) {
    console.error("❌ Error:", err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
