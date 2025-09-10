// AI Model Global Variables
let lstmModel = null;
let feedforwardModel = null;
const LSTM_THRESHOLD = 2000; // Minimum data points for LSTM
const FEEDFORWARD_THRESHOLD = 50; // Minimum data points for Feedforward

// Generate prediction using AI models
function generatePrediction() {
    const dataCount = fetchedData.length;
    let predictionValue;

    if (dataCount >= LSTM_THRESHOLD && lstmModel) {
        showNotification('Using LSTM AI Model...', 'info');
        const recentNumbers = fetchedData.slice(0, 10).map(item => parseInt(item.number) / 9);
        const inputTensor = tf.tensor2d([recentNumbers], [1, 10]);
        const predictionTensor = lstmModel.predict(inputTensor);
        predictionValue = predictionTensor.dataSync()[0];
    } else if (dataCount >= FEEDFORWARD_THRESHOLD && feedforwardModel) {
        showNotification('Using Feedforward AI Model...', 'info');
        const lastNumber = parseInt(fetchedData[0].number) / 9;
        const inputTensor = tf.tensor2d([[lastNumber]], [1, 1]);
        const predictionTensor = feedforwardModel.predict(inputTensor);
        predictionValue = predictionTensor.dataSync()[0];
    } else {
        showNotification('Using Rule-Based Prediction...', 'warning');
        if (dataCount < 10) {
            return Math.floor(Math.random() * 10);
        }
        const recentNumbers = fetchedData.slice(0, 10).map(item => parseInt(item.number));
        const lastNumber = recentNumbers[0];
        const secondLastNumber = recentNumbers[1];
        const prediction = (lastNumber + secondLastNumber) % 10;
        return prediction;
    }

    return Math.round(predictionValue * 9);
}

// Train AI models
async function trainAIModels() {
    const dataCount = fetchedData.length;

    // Train LSTM model if threshold is met and model is not yet trained
    if (dataCount >= LSTM_THRESHOLD && !lstmModel) {
        showNotification('Training LSTM AI model...', 'info');
        
        const data = fetchedData.slice(0, LSTM_THRESHOLD).map(item => parseInt(item.number));
        const normalizedData = data.map(num => num / 9);
        
        const xs = [];
        const ys = [];
        for (let i = 0; i < normalizedData.length - 10; i++) {
            const sequence = normalizedData.slice(i, i + 10);
            const label = normalizedData[i + 10];
            xs.push(sequence);
            ys.push(label);
        }
        
        const inputTensor = tf.tensor2d(xs, [xs.length, 10]);
        const outputTensor = tf.tensor2d(ys, [ys.length, 1]);

        const model = tf.sequential();
        model.add(tf.layers.lstm({ units: 50, inputShape: [10, 1], returnSequences: true }));
        model.add(tf.layers.lstm({ units: 50, returnSequences: false }));
        model.add(tf.layers.dense({ units: 1, activation: 'relu' }));
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        try {
            await model.fit(inputTensor, outputTensor, { epochs: 100, batchSize: 32 });
            lstmModel = model;
            showNotification('LSTM model trained and ready! 🎉', 'success');
            await lstmModel.save('localstorage://lstm-prediction-model');
        } catch (e) {
            showNotification('Error training LSTM model. Check console.', 'error');
            console.error(e);
        }
    }

    // Train Feedforward model if threshold is met and model is not yet trained
    if (dataCount >= FEEDFORWARD_THRESHOLD && !feedforwardModel) {
        showNotification('Training Feedforward AI model...', 'info');

        const data = fetchedData.slice(0, dataCount).map(item => parseInt(item.number));
        const normalizedData = data.map(num => num / 9);

        const xs = [];
        const ys = [];
        for (let i = 0; i < normalizedData.length - 1; i++) {
            xs.push([normalizedData[i]]);
            ys.push([normalizedData[i+1]]);
        }

        const inputTensor = tf.tensor2d(xs, [xs.length, 1]);
        const outputTensor = tf.tensor2d(ys, [ys.length, 1]);

        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 16, inputShape: [1], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'relu' }));
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        try {
            await model.fit(inputTensor, outputTensor, { epochs: 200, batchSize: 8 });
            feedforwardModel = model;
            showNotification('Feedforward model trained and ready!', 'success');
            await feedforwardModel.save('localstorage://feedforward-prediction-model');
        } catch (e) {
            showNotification('Error training Feedforward model. Check console.', 'error');
            console.error(e);
        }
    }
}

// Load AI models from local storage
async function loadAIModels() {
    try {
        const models = await tf.io.listModels();
        if ('localstorage://lstm-prediction-model' in models) {
            lstmModel = await tf.loadLayersModel('localstorage://lstm-prediction-model');
            showNotification('LSTM model loaded from storage. Ready!', 'info');
        }
        if ('localstorage://feedforward-prediction-model' in models) {
            feedforwardModel = await tf.loadLayersModel('localstorage://feedforward-prediction-model');
            showNotification('Feedforward model loaded from storage. Ready!', 'info');
        }
    } catch (e) {
        console.log('No AI models found in local storage.');
    }
}
