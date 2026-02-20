// Document Classifier Interactive JavaScript

// Example texts
const examples = {
    short: "stocks",
    long: "The global semiconductor industry is facing unprecedented demand as artificial intelligence applications proliferate across various sectors. Leading manufacturers like TSMC and Samsung are accelerating their production capacity to meet the needs of major tech firms. Meanwhile, governments in the US and Europe are providing significant subsidies to encourage domestic chip fabrication. This strategic shift aims to secure supply chains and gain a competitive edge in the next generation of computing. Experts believe that these investments will fundamentally transform the technological landscape for years to come.",
    ambiguous: "hospital CEO merger"
};

// DOM Elements
const inputText = document.getElementById('inputText');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const classifyBtn = document.getElementById('classifyBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Update threshold display
thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = e.target.value;
});

// Example buttons
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const exampleType = btn.dataset.example;
        inputText.value = examples[exampleType];
    });
});

// Clear button
clearBtn.addEventListener('click', () => {
    inputText.value = '';
    results.classList.add('hidden');
});

// Classify button
classifyBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();

    if (!text) {
        alert('Please enter some text to classify');
        return;
    }

    const threshold = parseFloat(thresholdSlider.value) / 100;

    // Show loading
    loading.classList.remove('hidden');
    results.classList.add('hidden');

    try {
        // Get CSRF token
        const csrftoken = getCookie('csrftoken');

        // Call API
        const response = await fetch('/classifier/api/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({
                text: text,
                threshold: threshold
            })
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Display results
        displayResults(data);

    } catch (error) {
        alert('Error classifying document: ' + error.message);
    } finally {
        loading.classList.add('hidden');
    }
});

// Auto-classify if requested
window.addEventListener('load', () => {
    const params = new URLSearchParams(window.location.search);
    const text = params.get('text');
    if (text && params.get('auto') === 'true') {
        const inputText = document.getElementById('inputText');
        if (inputText) {
            inputText.value = decodeURIComponent(text);
            document.getElementById('classifyBtn').click();
        }
    }
});

function displayResults(data) {
    // Show results section
    results.classList.remove('hidden');

    // Display badges
    displayBadges(data);

    // Display confidence message
    displayConfidenceMessage(data);

    // Display top features
    displayTopFeatures(data);

    // Display highlighted text
    displayHighlightedText(data);

    // Display preprocessing steps
    displayPreprocessingSteps(data);

    // Display all probabilities
    displayAllProbabilities(data);
}

function displayBadges(data) {
    const badgesContainer = document.getElementById('badges');
    const categoryCount = document.getElementById('categoryCount');

    const allCategories = Object.keys(data.all_probabilities);
    const predictedCount = data.predicted_labels.length;

    categoryCount.textContent = `Showing ${predictedCount} of ${allCategories.length} categories`;

    badgesContainer.innerHTML = '';

    // Sort by probability
    const sorted = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1]);

    sorted.forEach(([category, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const isPredicted = data.predicted_labels.includes(category);

        // Determine badge color
        let badgeClass = 'badge-gray';
        if (prob > 0.6) badgeClass = 'badge-green';
        else if (prob >= 0.3) badgeClass = 'badge-yellow';

        // Only show if predicted or above threshold
        if (isPredicted || prob >= parseFloat(thresholdSlider.value) / 100) {
            const badge = document.createElement('div');
            badge.className = `${badgeClass} text-white px-6 py-3 rounded-lg font-semibold shadow-lg flex items-center gap-2`;
            badge.innerHTML = `
                <span class="capitalize">${category}</span>
                <span class="text-sm opacity-90">${percentage}%</span>
            `;
            badgesContainer.appendChild(badge);
        }
    });
}

function displayConfidenceMessage(data) {
    const messageContainer = document.getElementById('confidenceMessage');

    // Calculate average confidence
    const avgConfidence = data.predicted_labels.reduce((sum, label) => {
        return sum + data.all_probabilities[label];
    }, 0) / data.predicted_labels.length;

    let icon, message, colorClass, borderClass;

    if (avgConfidence > 0.7) {
        icon = '🟢';
        message = 'Strong prediction';
        colorClass = 'confidence-high';
        borderClass = 'border-green-500';
    } else if (avgConfidence >= 0.4) {
        icon = '🟡';
        message = 'Reasonable prediction';
        colorClass = 'confidence-medium';
        borderClass = 'border-yellow-500';
    } else {
        icon = '🔴';
        message = 'Uncertain - may need more data';
        colorClass = 'confidence-low';
        borderClass = 'border-red-500';
    }

    messageContainer.className = `mb-6 p-4 rounded-lg border-l-4 ${borderClass} bg-gray-50`;
    messageContainer.innerHTML = `
        <div class="flex items-center gap-3">
            <span class="text-2xl">${icon}</span>
            <div>
                <div class="font-semibold ${colorClass}">${data.confidence_level.toUpperCase()} CONFIDENCE</div>
                <div class="text-sm text-gray-600">${message}</div>
            </div>
        </div>
    `;
}

function displayTopFeatures(data) {
    const featuresContainer = document.getElementById('topFeatures');
    featuresContainer.innerHTML = '';

    if (data.top_features && data.top_features.length > 0) {
        data.top_features.slice(0, 5).forEach((feature, index) => {
            const featureTag = document.createElement('div');
            featureTag.className = 'bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2';

            // Handle both old string format and new object format
            const word = typeof feature === 'object' ? feature.word : feature;
            const score = typeof feature === 'object' ? feature.score.toFixed(4) : null;

            featureTag.innerHTML = `
                <span>${index + 1}. <strong>${word}</strong></span>
                ${score ? `<span class="text-xs bg-purple-200 px-2 py-0.5 rounded-full">${score}</span>` : ''}
            `;
            featuresContainer.appendChild(featureTag);
        });
    } else {
        featuresContainer.innerHTML = '<span class="text-gray-500 text-sm">No features available</span>';
    }
}

function displayHighlightedText(data) {
    const highlightedContainer = document.getElementById('highlightedText');

    let highlightedText = data.text;

    // Highlight top features
    if (data.top_features && data.top_features.length > 0) {
        data.top_features.slice(0, 5).forEach(feature => {
            const word = typeof feature === 'object' ? feature.word : feature;
            // Create case-insensitive regex
            const regex = new RegExp(`\\b(${escapeRegex(word)})\\b`, 'gi');
            highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
        });
    }

    highlightedContainer.innerHTML = highlightedText;
}

function displayPreprocessingSteps(data) {
    console.log('Preprocessing steps received:', data.preprocessing_steps);
    const steps = data.preprocessing_steps;

    if (steps) {
        // Helper to find step by name (improved matching)
        const findStep = (name) => {
            const stepEntry = Object.values(steps).find(s =>
                s.name.toLowerCase().includes(name.toLowerCase()) ||
                name.toLowerCase().includes(s.name.toLowerCase())
            );
            return stepEntry ? stepEntry.result : null;
        };

        const original = findStep('Original');
        const lowercased = findStep('Lowercase');
        const stopWords = findStep('Tokenization');
        const lemmatized = findStep('Lemmatization') || findStep('Final') || findStep('Step5');

        if (original) document.getElementById('step1').textContent = original;
        if (lowercased) document.getElementById('step2').textContent = lowercased;
        if (stopWords) document.getElementById('step4').textContent = stopWords;
        // Step 5 - fallback to the very last step if lemmatization not found explicitly
        const lastStep = Object.values(steps)[Object.values(steps).length - 1].result;
        if (document.getElementById('step5')) {
            document.getElementById('step5').textContent = lemmatized || lastStep;
        }
    } else {
        // Fallback
        console.warn('No preprocessing steps in response data');
        document.getElementById('step1').textContent = data.text.substring(0, 200) + (data.text.length > 200 ? '...' : '');
        document.getElementById('step2').textContent = data.text.toLowerCase().substring(0, 200) + (data.text.length > 200 ? '...' : '');
        document.getElementById('step4').textContent = data.preprocessed_text || 'N/A';
        document.getElementById('step5').textContent = data.preprocessed_text || 'N/A';
    }
}

function displayAllProbabilities(data) {
    const probContainer = document.getElementById('allProbabilities');
    probContainer.innerHTML = '';

    // Sort by probability
    const sorted = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1]);

    sorted.forEach(([category, prob]) => {
        const percentage = (prob * 100).toFixed(2);
        const barWidth = prob * 100;

        const probRow = document.createElement('div');
        probRow.className = 'bg-gray-50 p-3 rounded-lg';
        probRow.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="font-semibold capitalize">${category}</span>
                <span class="text-sm text-gray-600">${percentage}%</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="bg-purple-600 h-2 rounded-full" style="width: ${barWidth}%"></div>
            </div>
        `;
        probContainer.appendChild(probRow);
    });
}

// Helper function to get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Helper function to escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
