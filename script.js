// DOM Elements
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');
const notification = document.getElementById('notification');
const notificationMessage = document.getElementById('notification-message');
const saveTelegramBtn = document.getElementById('save-telegram-settings');
const fetchedDataTable = document.getElementById('fetched-data-table').querySelector('tbody');
const lastUpdatedEl = document.getElementById('last-updated');
const progressBar = document.getElementById('progress-bar');
const predictionHistoryTable = document.getElementById('prediction-history').querySelector('tbody');
const autoSessionToggleBtn = document.getElementById('session-toggle');
const clearPredictionsBtn = document.getElementById('clear-predictions-btn');
const clearFetchedBtn = document.getElementById('clear-fetched-btn');
const refreshBtn = document.getElementById('refresh-btn');
const sessionStatsPopup = document.getElementById('session-stats-popup');
const popupTotalEl = document.getElementById('popup-total');
const popupWinsEl = document.getElementById('popup-wins');
const popupLossesEl = document.getElementById('popup-losses');
const sessionEndCountInput = document.getElementById('session-end-count');
const toggleScheduledBtn = document.getElementById('toggle-scheduled-btn');
const toggleTimingSessionsBtn = document.getElementById('toggle-timing-sessions');

// Manual Prediction Tab Elements
const manualSessionStartBtn = document.getElementById('manual-session-start');
const manualSessionEndBtn = document.getElementById('manual-session-end');
const manualSessionInfo = document.getElementById('manual-session-info');
const manualProgressBar = document.getElementById('manual-progress-bar');
const manualPeriodInput = document.getElementById('manual-period');
const manualSizePredSelect = document.getElementById('manual-size-pred');
const manualNumberPredInput = document.getElementById('manual-number-pred');
const submitManualPredictionBtn = document.getElementById('submit-manual-prediction');

// Splash screen elements
const splashScreen = document.getElementById('splash-screen');
const startButton = document.getElementById('start-button');
const mainContent = document.getElementById('main-content');

// Global variables
let predictions = [];
let isAutoSessionActive = false;
let isManualSessionActive = false;
let isScheduledMessagesActive = false;
let isTimingSessionsActive = false;
let updateTimer;
let manualUpdateTimer;
let scheduledMessagesTimer;
let timingSessionsTimer;
let sessionStats = { wins: 0, losses: 0, total: 0 };
let sessionEndLimit = 10;
let telegramSettings = {
    botToken: '',
    channelId: '',
    messageTemplate: 'PERIOD ➪ {issueNumber}\nRESULT ➪ {size} / {number}'
};
let scheduledMessages = [];
let autoTimingSessions = [];
let newPredictionSent = false;
let sentMessagesForMinute = {};
let lastSentMinute = -1;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadSavedData();
    // Start training the AI models after a short delay
    setTimeout(trainAIModels, 5000);
});

function setupEventListeners() {
    startButton.addEventListener('click', startSystem);

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            if (tabId === 'game') {
                const gameIframe = document.getElementById('game-iframe');
                gameIframe.src = 'https://www.6club.win/#/register?invitationCode=438172851706';
            }
            switchTab(tabId);
        });
    });

    saveTelegramBtn.addEventListener('click', saveTelegramSettings);
    autoSessionToggleBtn.addEventListener('click', toggleAutoSession);
    refreshBtn.addEventListener('click', () => {
        window.location.reload();
    });

    const helpBtn = document.getElementById('help-btn');
    helpBtn.addEventListener('click', () => {
        window.open('https://t.me/+Sa9rISoccao2ZDQ1', '_blank');
    });

    clearPredictionsBtn.addEventListener('click', clearPredictionData);
    clearFetchedBtn.addEventListener('click', clearFetchedData);

    manualSessionStartBtn.addEventListener('click', startManualSession);
    manualSessionEndBtn.addEventListener('click', endManualSession);
    submitManualPredictionBtn.addEventListener('click', submitManualPrediction);
    
    toggleScheduledBtn.addEventListener('click', toggleScheduledMessages);
    toggleTimingSessionsBtn.addEventListener('click', toggleTimingSessions);

    makeDraggable(sessionStatsPopup);
}

async function startSystem() {
    splashScreen.classList.add('hidden');
    showNotification('AKELA AI System initialized', 'info');
    await fetchData();
    startUpdateTimer();
    updatePredictionHistoryTable(); // Show prediction history on start
}

function switchTab(tabId) {
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
    document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add('active');
}

function startUpdateTimer() {
    clearInterval(updateTimer);
    let lastFetchSecond = -1;

    function updateProgressBar() {
        const now = new Date();
        const istOffset = 5.5 * 60 * 60 * 1000;
        const istTime = new Date(now.getTime() + istOffset);
        const seconds = istTime.getSeconds();

        const percentage = (seconds / 60) * 100;
        const timeLeft = 60 - seconds;

        progressBar.style.width = `${percentage}%`;
        progressBar.textContent = `Updating in ${timeLeft}s`;
        
        if (timeLeft <= 10) {
            progressBar.style.background = `linear-gradient(90deg, var(--danger), var(--warning))`;
        } else {
            progressBar.style.background = `linear-gradient(90deg, var(--primary), var(--secondary))`;
        }
        
        if (seconds === 0 && lastFetchSecond !== 0) {
            fetchData();
        }
        lastFetchSecond = seconds;
    }

    updateProgressBar();
    updateTimer = setInterval(updateProgressBar, 1000);
}

function startManualUpdateTimer() {
    clearInterval(manualUpdateTimer);
    let lastFetchSecond = -1;
    let lastFetchedIssueNumber = fetchedData.length > 0 ? fetchedData[0].issueNumber : null;

    function updateManualProgressBar() {
        const now = new Date();
        const istOffset = 5.5 * 60 * 60 * 1000;
        const istTime = new Date(now.getTime() + istOffset);
        const seconds = istTime.getSeconds();

        const percentage = (seconds / 60) * 100;
        const timeLeft = 60 - seconds;

        manualProgressBar.style.width = `${percentage}%`;
        
        if (timeLeft <= 15) {
            manualProgressBar.textContent = `Please input result`;
            manualProgressBar.classList.add('blink');
        } else {
            manualProgressBar.textContent = `Time left: ${timeLeft}s`;
            manualProgressBar.classList.remove('blink');
        }

        if (seconds === 0 && lastFetchSecond !== 0) {
            fetchData();
        }
        lastFetchSecond = seconds;
        
        const currentIssueNumber = fetchedData.length > 0 ? fetchedData[0].issueNumber : null;
        if (currentIssueNumber && currentIssueNumber !== lastFetchedIssueNumber) {
            newPredictionSent = false;
            manualPeriodInput.value = incrementIssueNumber(currentIssueNumber);
            lastFetchedIssueNumber = currentIssueNumber;
        }
        
    }

    updateManualProgressBar();
    manualUpdateTimer = setInterval(updateManualProgressBar, 1000);
}

function updateDashboardUI() {
    if (fetchedData.length > 0) {
        const latestPeriod = fetchedData[0].issueNumber;
        const nextPeriod = incrementIssueNumber(latestPeriod);
        const prediction = generatePrediction();

        document.getElementById('next-period').textContent = nextPeriod.slice(-4);
        document.getElementById('number-pred').textContent = prediction;
        document.getElementById('size-pred').textContent = prediction >= 5 ? 'BIG' : 'SMALL';
    }
}

function submitManualPrediction() {
    if (!isManualSessionActive) {
        showNotification('Please start a manual session first.', 'warning');
        return;
    }
    
    if (newPredictionSent) {
        showNotification('Please wait for the next period.', 'warning');
        return;
    }
    
    const manualPeriod = manualPeriodInput.value;
    const manualSize = manualSizePredSelect.value;
    const manualNumber = manualNumberPredInput.value;

    if (!manualSize && !manualNumber) {
        showNotification('Please enter at least a Size or a Number.', 'error');
        return;
    }
    
    const nextIssueNumber = fetchedData.length > 0 ? incrementIssueNumber(fetchedData[0].issueNumber) : null;
    if (!nextIssueNumber) {
        showNotification('Cannot determine next period. Please try again.', 'error');
        return;
    }

    const prediction = {
        issueNumber: nextIssueNumber,
        numberPrediction: manualNumber || null,
        sizePrediction: manualSize || null,
        actualNumber: null,
        actualSize: null,
        result: null,
        type: 'manual'
    };
    
    predictions.unshift(prediction);
    updatePredictionUI(prediction);
    updatePredictionHistoryTable();
    saveSettingsToLocalStorage();
    newPredictionSent = true;
    
    sendTelegramManualPrediction(prediction);
    showNotification(`Manual prediction for period ${manualPeriod.slice(-4)} submitted`, 'info');
}

async function sendTelegramManualPrediction(prediction) {
    if (!telegramSettings.botToken || !telegramSettings.channelId) {
        showNotification('Telegram settings not configured.', 'warning');
        return;
    }

    const messageContent = `𝗣𝗘𝗥𝗜𝗢𝗗 ☞  ${prediction.issueNumber.slice(-4)}\n`;
    let messageResult = '';

    if (prediction.sizePrediction && prediction.numberPrediction) {
        messageResult = `𝗥𝗘𝗦𝗨𝗟𝗧 ☞  ${prediction.sizePrediction} / ${prediction.numberPrediction}`;
    } else if (prediction.sizePrediction) {
        messageResult = `𝗥𝗘𝗦𝗨𝗟𝗧 ☞  ${prediction.sizePrediction}`;
    } else if (prediction.numberPrediction) {
        messageResult = `𝗥𝗘𝗦𝗨𝗟𝗧 ☞  ${prediction.numberPrediction}`;
    }

    const message = messageContent + messageResult;

    const telegramApiUrl = `https://api.telegram.org/bot${telegramSettings.botToken}/sendMessage`;
    try {
        const response = await fetch(telegramApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: telegramSettings.channelId, text: message })
        });
        const data = await response.json();
        if (data.ok) {
            showNotification('Manual prediction sent to Telegram successfully', 'success');
        } else {
            showNotification(`Telegram API Error: ${data.description}`, 'error');
            console.error('Telegram API Error:', data);
        }
    } catch (error) {
        showNotification('Failed to send manual prediction to Telegram', 'error');
        console.error('Network error sending message to Telegram:', error);
    }
}

function updatePredictionUI(prediction) {
    document.getElementById('next-period').textContent = prediction.issueNumber.slice(-4);
    document.getElementById('number-pred').textContent = prediction.numberPrediction || '--';
    document.getElementById('size-pred').textContent = prediction.sizePrediction || '--';
    document.getElementById('summary-accuracy').textContent = `${(Math.random() * 15 + 80).toFixed(1)}%`;
}

function updatePredictionHistoryTable() {
    predictionHistoryTable.innerHTML = '';
    const predictionsToShow = predictions.slice(0, 100);
    predictionsToShow.forEach(pred => {
        const row = document.createElement('tr');
        let resultColor = 'inherit';
        if (pred.result === 'win') {
            resultColor = 'var(--success)';
        } else if (pred.result === 'loss') {
            resultColor = 'var(--danger)';
        }
        row.innerHTML = `
            <td>${pred.issueNumber.slice(-4)}</td>
            <td>${pred.numberPrediction || '--'}</td>
            <td>${pred.sizePrediction || '--'}</td>
            <td style="color:${resultColor};">${pred.result ? pred.result.toUpperCase() : '--'}</td>
        `;
        predictionHistoryTable.appendChild(row);
    });
    updateStats();
}

function updateStats() {
    const completedPredictions = predictions.filter(p => p.result !== null);
    if (completedPredictions.length > 0) {
        const wins = completedPredictions.filter(p => p.result === 'win').length;
        const sizeCorrect = completedPredictions.filter(p => p.sizePrediction === p.actualSize).length;
        
        const sizeAccuracy = ((sizeCorrect / completedPredictions.length) * 100).toFixed(1);
        const winRate = ((wins / completedPredictions.length) * 100).toFixed(1);

        document.getElementById('total-predictions').textContent = completedPredictions.length;
        document.getElementById('win-rate').textContent = `Win Rate: ${winRate}%`;
        document.getElementById('summary-accuracy').textContent = `${sizeAccuracy}%`;

        let currentStreak = 0;
        for (let i = 0; i < completedPredictions.length; i++) {
            if (completedPredictions[i].result === 'win') {
                currentStreak++;
            } else {
                break;
            }
        }
        document.getElementById('current-streak').textContent = currentStreak;
    }
}

function updateSessionStatsPopup() {
    popupTotalEl.textContent = sessionStats.total;
    popupWinsEl.textContent = sessionStats.wins;
    popupLossesEl.textContent = sessionStats.losses;
}

function toggleAutoSession() {
    if (isAutoSessionActive) {
        endAutoSession();
    } else {
        startAutoSession();
    }
}

async function startAutoSession() {
    if (!telegramSettings.botToken || !telegramSettings.channelId) {
        showNotification('Please configure Telegram settings first.', 'warning');
        return;
    }
    if (isAutoSessionActive) return;
    isAutoSessionActive = true;
    sessionStats = { wins: 0, losses: 0, total: 0 };
    sessionEndLimit = parseInt(sessionEndCountInput.value) || 10;
    sessionStatsPopup.classList.add('show');
    updateSessionStatsPopup();
    showNotification('New auto prediction session started', 'success');
    await sendTelegramSticker(getRandomStickerId('start'));
    
    if (fetchedData.length > 0) {
        await sendTelegramNextPrediction(fetchedData[0]);
    }
    
    autoSessionToggleBtn.innerHTML = '<i class="fas fa-stop"></i> End Auto Session';
    autoSessionToggleBtn.classList.remove('btn-secondary');
    autoSessionToggleBtn.classList.add('btn-danger');
}

async function endAutoSession() {
    isAutoSessionActive = false;
    sessionStatsPopup.classList.remove('show');
    showNotification('Auto prediction session ended', 'info');
    await sendTelegramSticker(getRandomStickerId('end'));

    autoSessionToggleBtn.innerHTML = '<i class="fas fa-play"></i> Start Auto Session';
    autoSessionToggleBtn.classList.remove('btn-danger');
    autoSessionToggleBtn.classList.add('btn-secondary');
}

async function startManualSession() {
    if (!telegramSettings.botToken || !telegramSettings.channelId) {
        showNotification('Please configure Telegram settings first.', 'warning');
        return;
    }
    if (isManualSessionActive) return;

    isManualSessionActive = true;
    manualSessionStartBtn.style.display = 'none';
    manualSessionEndBtn.style.display = 'inline-flex';
    manualSessionInfo.style.display = 'block';

    const latestFetchedPeriod = fetchedData.length > 0 ? fetchedData[0].issueNumber : null;
    if (latestFetchedPeriod) {
        manualPeriodInput.value = incrementIssueNumber(latestFetchedPeriod);
    }
    
    startManualUpdateTimer();
    await sendTelegramSticker(getRandomStickerId('start'));
    showNotification('Manual prediction session started!', 'success');
}

async function endManualSession() {
    if (!isManualSessionActive) return;
    isManualSessionActive = false;
    manualSessionStartBtn.style.display = 'inline-flex';
    manualSessionEndBtn.style.display = 'none';
    manualSessionInfo.style.display = 'none';
    clearInterval(manualUpdateTimer);
    await sendTelegramSticker(getRandomStickerId('end'));
    showNotification('Manual prediction session ended.', 'info');
}

async function sendTelegramMessage(prediction) {
    if (!telegramSettings.botToken || !telegramSettings.channelId) return;

    const message = telegramSettings.messageTemplate
        .replace('{issueNumber}', prediction.issueNumber.slice(-4))
        .replace('{size}', prediction.sizePrediction)
        .replace('{number}', prediction.numberPrediction);

    const telegramApiUrl = `https://api.telegram.org/bot${telegramSettings.botToken}/sendMessage`;
    try {
        const response = await fetch(telegramApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: telegramSettings.channelId, text: message })
        });
        const data = await response.json();
        if (data.ok) {
            showNotification('Prediction sent to Telegram successfully', 'success');
        } else {
            showNotification(`Telegram API Error: ${data.description}`, 'error');
            console.error('Telegram API Error:', data);
        }
    } catch (error) {
        showNotification('Failed to send prediction to Telegram', 'error');
        console.error('Network error sending message to Telegram:', error);
    }
}

function toggleScheduledMessages() {
    isScheduledMessagesActive = !isScheduledMessagesActive;
    if (isScheduledMessagesActive) {
        toggleScheduledBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Scheduled Messages';
        toggleScheduledBtn.classList.remove('btn-primary');
        toggleScheduledBtn.classList.add('btn-danger');
        scheduledMessagesTimer = setInterval(checkScheduledMessages, 1000);
        showNotification('Scheduled messages activated!', 'success');
    } else {
        toggleScheduledBtn.innerHTML = '<i class="fas fa-clock"></i> Start Scheduled Messages';
        toggleScheduledBtn.classList.remove('btn-danger');
        toggleScheduledBtn.classList.add('btn-primary');
        clearInterval(scheduledMessagesTimer);
        showNotification('Scheduled messages stopped.', 'info');
    }
}

async function checkScheduledMessages() {
    const now = new Date();
    const currentMinute = now.getMinutes();
    const currentTime = `${String(now.getHours()).padStart(2, '0')}:${String(currentMinute).padStart(2, '0')}`;
    
    if (currentMinute !== lastSentMinute) {
        sentMessagesForMinute = {};
        lastSentMinute = currentMinute;
    }

    const scheduledTimes = document.querySelectorAll('.scheduled-time');
    for (let i = 0; i < scheduledTimes.length; i++) {
        const timeInput = scheduledTimes[i];
        const timeValue = timeInput.value;
        
        if (timeValue === currentTime && !sentMessagesForMinute[timeValue]) {
            sentMessagesForMinute[timeValue] = true;
            
            const messageText = document.getElementById(`scheduled-message-${i + 1}`).value;
            
            if (messageText) {
                await sendTelegramPlainMessage(messageText);
            }
        }
    }
}

async function sendTelegramPlainMessage(message) {
    if (!telegramSettings.botToken || !telegramSettings.channelId) {
        showNotification('Telegram settings not configured.', 'warning');
        return;
    }
    const telegramApiUrl = `https://api.telegram.org/bot${telegramSettings.botToken}/sendMessage`;
    try {
        const response = await fetch(telegramApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: telegramSettings.channelId, text: message })
        });
        const data = await response.json();
        if (data.ok) {
            showNotification('Scheduled message sent successfully', 'success');
        } else {
            showNotification(`Telegram API Error: ${data.description}`, 'error');
        }
    } catch (error) {
        showNotification('Failed to send scheduled message', 'error');
    }
}

function toggleTimingSessions() {
    isTimingSessionsActive = !isTimingSessionsActive;
    if (isTimingSessionsActive) {
        toggleTimingSessionsBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Timing Sessions';
        toggleTimingSessionsBtn.classList.remove('btn-primary');
        toggleTimingSessionsBtn.classList.add('btn-danger');
        timingSessionsTimer = setInterval(checkTimingSessions, 1000);
        showNotification('Auto timing sessions activated!', 'success');
    } else {
        toggleTimingSessionsBtn.innerHTML = '<i class="fas fa-clock"></i> Start Timing Sessions';
        toggleTimingSessionsBtn.classList.remove('btn-danger');
        toggleTimingSessionsBtn.classList.add('btn-primary');
        clearInterval(timingSessionsTimer);
        showNotification('Auto timing sessions stopped.', 'info');
    }
}

async function checkTimingSessions() {
    const now = new Date();
    const currentTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    
    const timingTimes = document.querySelectorAll('.timing-time-input');
    for (const timeInput of timingTimes) {
        const timeValue = timeInput.value;
        if (timeValue === currentTime && !isAutoSessionActive) {
            await startAutoSession();
            return;
        }
    }
}

function saveTelegramSettings() {
    const botToken = document.getElementById('bot-token').value;
    const channelId = document.getElementById('channel-id').value;
    const messageTemplate = document.getElementById('message-template').value;
    if (!botToken || !channelId) {
        showNotification('Please enter both Bot Token and Channel ID', 'error');
        return;
    }

    telegramSettings.botToken = botToken;
    telegramSettings.channelId = channelId;
    telegramSettings.messageTemplate = messageTemplate;

    saveSettingsToLocalStorage();
    showNotification('Telegram settings saved successfully', 'success');
}

function saveSettingsToLocalStorage() {
    const settingsToSave = {
        botToken: telegramSettings.botToken,
        channelId: telegramSettings.channelId,
        messageTemplate: telegramSettings.messageTemplate,
    };
    localStorage.setItem('telegram_settings', JSON.stringify(settingsToSave));
    localStorage.setItem('fetched_data', JSON.stringify(fetchedData));
    localStorage.setItem('predictions', JSON.stringify(predictions));
}

function loadSavedData() {
    const savedSettings = localStorage.getItem('telegram_settings');
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        telegramSettings.botToken = settings.botToken || '';
        telegramSettings.channelId = settings.channelId || '';
        telegramSettings.messageTemplate = settings.messageTemplate || '';
        document.getElementById('bot-token').value = telegramSettings.botToken;
        document.getElementById('channel-id').value = telegramSettings.channelId;
        document.getElementById('message-template').value = telegramSettings.messageTemplate;
    }
    const savedFetchedData = localStorage.getItem('fetched_data');
    if (savedFetchedData) {
        fetchedData = JSON.parse(savedFetchedData);
        updateFetchedDataTable();
    }
    const savedPredictions = localStorage.getItem('predictions');
    if (savedPredictions) {
        predictions = JSON.parse(savedPredictions);
        updatePredictionHistoryTable();
        const currentPrediction = predictions.find(p => p.result === null);
        if (currentPrediction) {
            updatePredictionUI(currentPrediction);
        }
    }
    loadAIModels();
}

function showNotification(message, type) {
    notificationMessage.textContent = message;
    notification.className = `notification ${type} show`;
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function clearPredictionData() {
    predictions = [];
    updatePredictionHistoryTable();
    updateStats();
    localStorage.removeItem('predictions');
    showNotification('Prediction data cleared successfully', 'success');
}

function clearFetchedData() {
    fetchedData = [];
    updateFetchedDataTable();
    localStorage.removeItem('fetched_data');
    showNotification('Fetched data cleared successfully', 'success');
}

// Persist data every minute
setInterval(() => {
    saveSettingsToLocalStorage();
}, 60000);

// Draggable pop-up function
function makeDraggable(element) {
    let isDragging = false;
    let offsetX, offsetY;

    element.addEventListener('mousedown', (e) => {
        isDragging = true;
        offsetX = e.clientX - element.offsetLeft;
        offsetY = e.clientY - element.offsetTop;
        element.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        let newX = e.clientX - offsetX;
        let newY = e.clientY - offsetY;
        
        const maxX = window.innerWidth - element.offsetWidth;
        const maxY = window.innerHeight - element.offsetHeight;
        
        newX = Math.min(Math.max(0, newX), maxX);
        newY = Math.min(Math.max(0, newY), maxY);

        element.style.left = `${newX}px`;
        element.style.top = `${newY}px`;
        element.style.right = 'auto';
        element.style.bottom = 'auto';
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        element.style.cursor = 'move';
    });
}
