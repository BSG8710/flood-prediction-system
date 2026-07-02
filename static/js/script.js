// ==========================================================================
// FloodGuard front-end behaviour: mobile nav toggle + prediction form
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupMobileNav();
    setupPredictionForm();
});

function setupMobileNav() {
    const toggle = document.getElementById('navToggle');
    const links = document.getElementById('navLinks');
    if (!toggle || !links) return;

    toggle.addEventListener('click', () => {
        links.classList.toggle('open');
    });
}

function setupPredictionForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return; // Not on the predict page

    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const formError = document.getElementById('formError');

    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');
    const riskBadge = document.getElementById('riskBadge');
    const riskMessage = document.getElementById('riskMessage');
    const floodBar = document.getElementById('floodBar');
    const noFloodBar = document.getElementById('noFloodBar');
    const floodPercent = document.getElementById('floodPercent');
    const noFloodPercent = document.getElementById('noFloodPercent');
    const modelName = document.getElementById('modelName');
    const resetBtn = document.getElementById('resetBtn');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        hideError();

        const formData = new FormData(form);
        const payload = {};
        for (const [key, value] of formData.entries()) {
            payload[key] = value;
        }

        // Basic client-side validation: every field must be a finite number.
        for (const [key, value] of Object.entries(payload)) {
            if (value === '' || Number.isNaN(Number(value))) {
                showError(`Please enter a valid number for "${key}".`);
                return;
            }
        }

        setLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || 'Something went wrong. Please try again.');
                return;
            }

            renderResult(data);
        } catch (err) {
            showError('Could not reach the prediction service. Check your connection and try again.');
        } finally {
            setLoading(false);
        }
    });

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            form.reset();
            resultContent.style.display = 'none';
            resultPlaceholder.style.display = 'block';
            form.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    }

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        btnText.textContent = isLoading ? 'Analyzing...' : 'Predict Flood Risk';
        btnSpinner.style.display = isLoading ? 'inline-block' : 'none';
    }

    function showError(message) {
        formError.textContent = message;
        formError.style.display = 'block';
    }

    function hideError() {
        formError.style.display = 'none';
        formError.textContent = '';
    }

    function renderResult(data) {
        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'block';

        riskBadge.className = `risk-badge ${data.risk_level}`;
        riskBadge.textContent = data.prediction === 1 ? 'Flood Risk Detected' : 'Low Risk';
        riskMessage.textContent = data.message;
        modelName.textContent = data.model_name;

        floodPercent.textContent = `${data.flood_probability}%`;
        noFloodPercent.textContent = `${data.no_flood_probability}%`;

        // Reset bars to 0 first so the fill animates on every submission.
        floodBar.style.width = '0%';
        noFloodBar.style.width = '0%';
        requestAnimationFrame(() => {
            setTimeout(() => {
                floodBar.style.width = `${data.flood_probability}%`;
                noFloodBar.style.width = `${data.no_flood_probability}%`;
            }, 50);
        });

        resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}
