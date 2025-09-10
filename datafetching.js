// Global variables for data fetching
let fetchedData = [];
let lastProcessedPeriod = null;

// Sticker IDs
const STICKER_IDS = {
    win_photo: [
        'CAACAgUAAxkBAAEGwtJovqkGOLQySXaXtspS_vnhCDydXwACfhUAAja6uVeIMbr1ctc3wR4E',
        'CAACAgUAAxkBAAEGwtBovqkR06Lhqe3waAuHDEeyaeJepgACIhIAAlFAyVZErDvM4_iAOR4E',
        'CAACAgUAAxkBAAEGws1ovqkeh0SEfVqBHxZSl0ODucl1kQACoBMAAqJxKFdkaJFPcIQNbR4E',
        'CAACAgUAAxkBAAEGwsZovqlYhIu9lTd59n-_9EbQpaXDiwAC1A8AAqdIiVW0E9TpqZEu8x4E',
        'CAACAgUAAxkBAAEGwsJovql1dZdwDXb3HkADfXZOb1nKLgACJxAAAoQ06Vfrak_IqecfsR4E',
        'CAACAgUAAxkBAAEGwtdovqfzJE-HmQmM_yx5do0QIFzt5wACXRwAAiYSaVQQiZvPI5keKh4E',
        'CAACAgUAAxkBAAEGwtZovqjzxv3Wa-zSOHmgfogidzojxwAC9hgAAstA8VUXBfVUCbQdYh4E',
        'CAACAgUAAxkBAAEGwpVovtgjHa3CxGr_JAtinRW9GszC9wACAhIAAknyKFYYnDtglztLNx4E',
        'CAACAgUAAxkBAAEGwpdovtgBl7HeMprK862bEoVuVZkXawACHRUAAsOn4Fc2h9u69xnxRx4E',
        'CAACAgUAAxkBAAEGwsFovqmB6jgSkGhBeQmBKWbVxHjk9wAC0hcAAsPycVSffCbGo2_kcB4E'
    ],
    win_video: [
        'CAACAgUAAxkBAAEGwqhovqqeM1R9IpH-ZxVnW9_d6LvGPQACUxMAAkPLSVVfxaN8acPr9h4E',
        'CAACAgUAAxkBAAEGwqFovsFcKygH6JfcqMD9Jb8xgOZznQACbhsAAk5QSFT5v5Iwn5EDox4E',
        'CAACAgUAAxkBAAEGwpZovtgEZ8QPylLyF78TsXuuL4QAAVkAAmwUAAJEuNFV165iAAHIXmheHgQ',
        'CAACAgUAAxkBAAEGw05ovuNmMLu11dmNcTi4UjhNYqqDNwACsxYAAiNJwFbspjV0MTl3Bx4E',
        'CAACAgUAAxkBAAEGw09ovuNnje_LTBQaYKNZyLHmwfskigAC2REAAvYw2FUsPYnqXf3MAh4E',
        'CAACAgUAAxkBAAEGw0hovuJ_hmAZLG05_25d3OCl7KtReQACthQAAq_OQVWeYIFhVYCDWR4E',
        'CAACAgUAAxkBAAEGwpVovtgjHa3CxGr_JAtinRW9GszC9wACAhIAAknyKFYYnDtglztLNx4E',
        'CAACAgUAAxkBAAEGwpZovtgEZ8QPylLyF78TsXuuL4QAAVkAAmwUAAJEuNFV165iAAHIXmheHgQ',
        'CAACAgUAAxkBAAEGwpdovtgBl7HeMprK862bEoVuVZkXawACHRUAAsOn4Fc2h9u69xnxRx4E',
        'CAACAgUAAxkBAAEGw0NovuJeSZMMMvyJ9twUgwmJzcseDQAC5BIAAhBE0VVZZYrWJBnXHR4E'
    ],
    start: ['CAACAgUAAxkBAAEC-dpov46ZJKtdi4aWHpFP1quggFt-DAACEBYAAgvYYFZw-0A5gsyL8B4E'],
    end: ['CAACAgUAAxkBAAEC-d1ov5AD8_MuWbPDpCSVC6OCEntAhgACVBMAAn5aYFbgPVLalZCklx4E'],
    loss: [
        'CAACAgUAAxkBAAEC-eJov5BHuSGlx2M7uS9f9gy0C-jsewACCBUAAmYFYFZM3xtfU1k19R4E',
        'CAACAgUAAxkBAAEC-eBov5BFTXICGlwyjLKTWdnMX_m4bQACwRIAAojOYVabOteK5QtLUx4E',
        'CAACAgUAAxkBAAEC-eFov5BG-yRGw93srk9QsgTKJvd5tQAC-A8AAgX_YVYeCYv6_XVApB4E'
    ]
};

// State for win sticker sequence
let lastWinPhotoIndex = -1;
let lastWinVideoIndex = -1;

// Fetch data from API
async function fetchData() {
    try {
        const timestamp = Date.now();
        const apiUrl = `https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?ts=${timestamp}`;
        const response = await fetch(apiUrl);
        const data = await response.json();
        
        if (data && data.data && data.data.list && data.data.list.length > 0) {
            await processNewData(data.data.list);
            updateDashboardUI();
        } else {
            showNotification('No data received from API', 'warning');
        }
    } catch (error) {
        console.error('Error fetching data:', error);
        showNotification('Error fetching data. Check console for details.', 'error');
    }
}

// Process new data from API
async function processNewData(newData) {
    const uniqueNewData = newData.filter(item => {
        return !fetchedData.some(existing => existing.issueNumber === item.issueNumber);
    });
    
    if (uniqueNewData.length === 0) {
        return;
    }
    
    const processedData = uniqueNewData.map(item => ({
        ...item,
        timestamp: new Date().toISOString(),
        size: parseInt(item.number) >= 5 ? 'BIG' : 'SMALL'
    }));
    
    fetchedData = [...processedData, ...fetchedData];
    if (fetchedData.length > 30000) {
        fetchedData = fetchedData.slice(0, 30000);
    }
    
    // Train AI models if data threshold is met
    trainAIModels();
    
    updateFetchedDataTable();
    updateLastUpdated();
    
    processedData.sort((a, b) => parseInt(a.issueNumber) - parseInt(b.issueNumber));
    
    for (const item of processedData) {
        await verifyAndSend(item);
    }
    
    showNotification(`Processed ${processedData.length} new records`, 'success');
}

// Update fetched data table
function updateFetchedDataTable() {
    fetchedDataTable.innerHTML = '';
    const dataToShow = fetchedData.slice(0, 100);
    dataToShow.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td title="${item.issueNumber}">${item.issueNumber.slice(-4)}</td>
            <td>${item.number}</td>
            <td>${item.size}</td>
        `;
        fetchedDataTable.appendChild(row);
    });
    document.getElementById('total-records').textContent = fetchedData.length;
}

// Update last updated timestamp
function updateLastUpdated() {
    lastUpdatedEl.textContent = `Last updated: ${new Date().toLocaleString()}`;
}

// Increment issue number
function incrementIssueNumber(issueNumber) {
    const lastFourDigits = issueNumber.slice(-4);
    const incremented = (parseInt(lastFourDigits) + 1).toString().padStart(4, '0');
    return issueNumber.slice(0, -4) + incremented;
}

// Verify prediction and send results
async function verifyAndSend(newestData) {
    const latestPrediction = predictions.find(p => p.issueNumber === newestData.issueNumber);
    if (latestPrediction && latestPrediction.result === null) {
        latestPrediction.actualNumber = newestData.number;
        latestPrediction.actualSize = newestData.size;
        
        const isSizeCorrect = latestPrediction.sizePrediction && latestPrediction.sizePrediction === newestData.size;
        const isNumberCorrect = latestPrediction.numberPrediction && parseInt(latestPrediction.numberPrediction) === parseInt(newestData.number);
        
        const isWin = isSizeCorrect || isNumberCorrect;
        latestPrediction.result = isWin ? 'win' : 'loss';
        updatePredictionHistoryTable();
        
        setTimeout(async () => {
            if (isWin) {
                await sendTelegramSticker(getRandomStickerId('win_photo'));
                await sendTelegramSticker(getRandomStickerId('win_video'));
            } else {
                await sendTelegramSticker(getRandomStickerId('loss'));
            }
            if (latestPrediction.type === 'auto') {
                if (isAutoSessionActive) {
                    sessionStats.total++;
                    if (isWin) {
                        sessionStats.wins++;
                    } else {
                        sessionStats.losses++;
                    }
                    updateSessionStatsPopup();
                    
                    if (sessionStats.wins >= sessionEndLimit) {
                        await endAutoSession();
                        return;
                    }
                    
                    await sendTelegramNextPrediction(newestData);
                }
            } else if (latestPrediction.type === 'manual') {
                newPredictionSent = false;
            }
        }, 10000); // 10-second delay
    } else {
        if (isAutoSessionActive) {
            await sendTelegramNextPrediction(newestData);
        }
    }
}

// Send next prediction to Telegram
async function sendTelegramNextPrediction(lastItem) {
    const nextIssueNumber = incrementIssueNumber(lastItem.issueNumber);
    const numberPrediction = generatePrediction();
    const sizePrediction = numberPrediction >= 5 ? 'BIG' : 'SMALL';
    
    const prediction = {
        timestamp: new Date().toISOString(),
        issueNumber: nextIssueNumber,
        numberPrediction,
        sizePrediction,
        actualNumber: null,
        actualSize: null,
        result: null,
        type: 'auto'
    };
    
    predictions.unshift(prediction);
    if (predictions.length > 30000) {
        predictions = predictions.slice(0, 30000);
    }
    
    updatePredictionUI(prediction);
    updatePredictionHistoryTable();
    
    if (telegramSettings.botToken && telegramSettings.channelId) {
        await sendTelegramMessage(prediction);
    }
}

function getRandomStickerId(type) {
    const stickers = STICKER_IDS[type];
    if (!stickers || stickers.length === 0) {
        return null;
    }
    
    if (type === 'win_photo' || type === 'win_video') {
        const lastIndexKey = `last_${type.replace('_', '')}Index`;
        let newIndex;
        do {
            newIndex = Math.floor(Math.random() * stickers.length);
        } while (newIndex === window[lastIndexKey]);
        window[lastIndexKey] = newIndex;
        return stickers[newIndex];
    } else {
        return stickers[Math.floor(Math.random() * stickers.length)];
    }
}

async function sendTelegramSticker(stickerId) {
    if (!telegramSettings.botToken || !telegramSettings.channelId || !stickerId) {
        return;
    }

    const telegramApiUrl = `https://api.telegram.org/bot${telegramSettings.botToken}/sendSticker`;
    const formData = new FormData();
    formData.append('chat_id', telegramSettings.channelId);
    formData.append('sticker', stickerId);

    try {
        const response = await fetch(telegramApiUrl, { method: 'POST', body: formData });
        const data = await response.json();
        if (data.ok) {
            showNotification('Sticker sent to Telegram successfully', 'success');
        } else {
            showNotification(`Telegram Sticker Error: ${data.description}`, 'error');
            console.error('Telegram Sticker Error:', data);
        }
    } catch (error) {
        showNotification('Failed to send sticker to Telegram', 'error');
    }
}
