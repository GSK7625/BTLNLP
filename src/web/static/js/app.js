document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const exampleSelect = document.getElementById('example-select');
    const contextInput = document.getElementById('context-input');
    const questionInput = document.getElementById('question-input');
    const goldAnswerGroup = document.getElementById('gold-answer-group');
    const goldInput = document.getElementById('gold-input');
    
    const checkBm25 = document.getElementById('model-bm25');
    const checkPretrained = document.getElementById('model-pretrained');
    const checkFinetuned = document.getElementById('model-finetuned');
    const checkQwen = document.getElementById('model-qwen');
    
    const btnSubmit = document.getElementById('btn-submit');
    const btnClear = document.getElementById('btn-clear');
    
    const resultsLoader = document.getElementById('results-loader');
    const emptyState = document.getElementById('empty-state');
    const resultsContent = document.getElementById('results-content');
    
    // Result cards
    const cardBm25 = document.getElementById('card-bm25');
    const cardPretrained = document.getElementById('card-pretrained');
    const cardFinetuned = document.getElementById('card-finetuned');
    const cardQwen = document.getElementById('card-qwen');
    
    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const highlightViewer = document.getElementById('highlight-viewer');
    
    // State variables
    let preloadedExamples = [];
    let currentResults = null;
    let activeTabModel = 'bm25'; // Default active tab

    // Initialize - Load preloaded examples
    fetch('/api/examples')
        .then(res => res.json())
        .then(data => {
            preloadedExamples = data;
            populateExamples(data);
        })
        .catch(err => console.error('Error loading examples:', err));

    // Helper: Escape HTML to prevent injection
    function escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Populate Examples dropdown
    function populateExamples(examples) {
        examples.forEach(ex => {
            const opt = document.createElement('option');
            opt.value = ex.id;
            // Shorten question to fit nicely
            const qShort = ex.question.length > 50 ? ex.question.substring(0, 50) + '...' : ex.question;
            opt.textContent = `[${ex.id.startsWith('ex_') ? 'Ví dụ' : 'Test'}] ${qShort}`;
            exampleSelect.appendChild(opt);
        });
    }

    // Handle example selection
    exampleSelect.addEventListener('change', (e) => {
        const selectedId = e.target.value;
        if (!selectedId) {
            // User chose custom input option
            contextInput.value = '';
            questionInput.value = '';
            goldInput.value = '';
            goldAnswerGroup.style.display = 'none';
            return;
        }
        
        const example = preloadedExamples.find(ex => ex.id === selectedId);
        if (example) {
            contextInput.value = example.context;
            questionInput.value = example.question;
            if (example.gold) {
                goldInput.value = example.gold;
                goldAnswerGroup.style.display = 'block';
            } else {
                goldInput.value = '';
                goldAnswerGroup.style.display = 'none';
            }
        }
    });

    // Handle submit/run
    btnSubmit.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        const context = contextInput.value.trim();
        const gold = goldInput.value.trim();
        
        if (!question || !context) {
            alert('Vui lòng nhập đầy đủ câu hỏi và đoạn ngữ cảnh.');
            return;
        }

        const models = [];
        if (checkBm25.checked) models.push('bm25');
        if (checkPretrained.checked) models.push('pretrained');
        if (checkFinetuned.checked) models.push('finetuned');
        if (checkQwen.checked) models.push('qwen');
        
        if (models.length === 0) {
            alert('Vui lòng chọn ít nhất một phương pháp để chạy.');
            return;
        }

        // Show loading state
        resultsLoader.style.display = 'flex';
        btnSubmit.disabled = true;
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, context, models, gold })
            });
            
            const data = await response.json();
            if (data.error) {
                alert('Có lỗi xảy ra: ' + data.error);
                return;
            }

            currentResults = data.results;
            
            // Show results UI
            emptyState.style.display = 'none';
            resultsContent.style.display = 'flex';
            
            // Update the metrics cards
            updateMetricCards(models, currentResults);
            
            // Auto switch active tab to one of the selected models
            if (models.includes(activeTabModel)) {
                // keep current active tab
            } else {
                activeTabModel = models[0];
            }
            
            // Update active state on tab buttons
            tabBtns.forEach(btn => {
                const model = btn.getAttribute('data-model');
                btn.style.display = models.includes(model) ? 'flex' : 'none';
                if (model === activeTabModel) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            // Render text highlighting
            renderHighlight(context, activeTabModel);
            
        } catch (err) {
            console.error(err);
            alert('Không thể kết nối đến server.');
        } finally {
            resultsLoader.style.display = 'none';
            btnSubmit.disabled = false;
        }
    });

    // Populate model metrics cards
    function updateMetricCards(selectedModels, results) {
        // Toggle card visibilities
        cardBm25.style.display = selectedModels.includes('bm25') ? 'flex' : 'none';
        cardPretrained.style.display = selectedModels.includes('pretrained') ? 'flex' : 'none';
        cardFinetuned.style.display = selectedModels.includes('finetuned') ? 'flex' : 'none';
        cardQwen.style.display = selectedModels.includes('qwen') ? 'flex' : 'none';
        
        // Populate BM25
        if (selectedModels.includes('bm25') && results.bm25) {
            const res = results.bm25;
            document.getElementById('ans-bm25').textContent = res.answer || '(Trống)';
            document.getElementById('lat-bm25').textContent = `${res.latency_ms}ms`;
            document.getElementById('conf-bm25').textContent = '1.000';
            updateGoldBadges('bm25', res);
        }
        
        // Populate Pretrained
        if (selectedModels.includes('pretrained') && results.pretrained) {
            const res = results.pretrained;
            document.getElementById('ans-pretrained').textContent = res.answer || '(Trống)';
            document.getElementById('lat-pretrained').textContent = `${res.latency_ms}ms`;
            document.getElementById('conf-pretrained').textContent = res.confidence !== undefined ? res.confidence.toFixed(3) : 'N/A';
            updateGoldBadges('pretrained', res);
        }
        
        // Populate Fine-tuned
        if (selectedModels.includes('finetuned') && results.finetuned) {
            const res = results.finetuned;
            document.getElementById('ans-finetuned').textContent = res.answer || '(Trống)';
            document.getElementById('lat-finetuned').textContent = `${res.latency_ms}ms`;
            document.getElementById('conf-finetuned').textContent = res.confidence !== undefined ? res.confidence.toFixed(3) : 'N/A';
            updateGoldBadges('finetuned', res);
        }

        // Populate Qwen
        if (selectedModels.includes('qwen') && results.qwen) {
            const res = results.qwen;
            document.getElementById('ans-qwen').textContent = res.answer || '(Trống)';
            document.getElementById('lat-qwen').textContent = `${res.latency_ms}ms`;
            document.getElementById('conf-qwen').textContent = res.confidence !== undefined ? res.confidence.toFixed(3) : 'N/A';
            updateGoldBadges('qwen', res);
        }
    }

    // Render F1 / EM metrics for gold standards
    function updateGoldBadges(modelKey, res) {
        const container = document.getElementById(`gold-metrics-${modelKey}`);
        const emBadge = document.getElementById(`em-${modelKey}`);
        const f1Badge = document.getElementById(`f1-${modelKey}`);
        
        if (res.em === null || res.f1 === null) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        
        // EM badge styling
        emBadge.textContent = `EM: ${res.em}`;
        emBadge.className = 'metric-badge';
        if (res.em === 1) {
            emBadge.classList.add('correct');
        } else {
            emBadge.classList.add('incorrect');
        }
        
        // F1 badge styling
        f1Badge.textContent = `F1: ${(res.f1 * 100).toFixed(1)}%`;
        f1Badge.className = 'metric-badge';
        if (res.f1 === 1.0) {
            f1Badge.classList.add('correct');
        } else if (res.f1 > 0.0) {
            f1Badge.classList.add('partial');
        } else {
            f1Badge.classList.add('incorrect');
        }
    }

    // Render highlighted context text
    function renderHighlight(context, modelKey) {
        if (!currentResults || !currentResults[modelKey]) {
            highlightViewer.textContent = context;
            return;
        }

        const res = currentResults[modelKey];
        if (res.error) {
            highlightViewer.innerHTML = `<span style="color:#ef4444;">[Lỗi Model] ${escapeHtml(res.error)}</span>`;
            return;
        }

        const start = res.char_start;
        const end = res.char_end;

        if (start === undefined || end === undefined || start < 0 || end < 0 || start > context.length || end > context.length || start > end) {
            // Fallback to exact text search if offsets are corrupt
            const ansText = res.answer;
            if (ansText && context.includes(ansText)) {
                const idx = context.indexOf(ansText);
                const before = context.substring(0, idx);
                const match = context.substring(idx, idx + ansText.length);
                const after = context.substring(idx + ansText.length);
                highlightViewer.innerHTML = escapeHtml(before) + 
                    `<span class="span-highlight hl-${modelKey}">${escapeHtml(match)}</span>` + 
                    escapeHtml(after);
            } else {
                highlightViewer.textContent = context;
            }
            return;
        }

        // Highlight using exact character indices
        const before = context.substring(0, start);
        const match = context.substring(start, end);
        const after = context.substring(end);

        highlightViewer.innerHTML = escapeHtml(before) + 
            `<span class="span-highlight hl-${modelKey}">${escapeHtml(match)}</span>` + 
            escapeHtml(after);
    }

    // Handle tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const modelKey = btn.getAttribute('data-model');
            activeTabModel = modelKey;
            
            // Toggle active state
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Rerender highlight
            const context = contextInput.value;
            renderHighlight(context, activeTabModel);
        });
    });

    // Clear form
    btnClear.addEventListener('click', () => {
        exampleSelect.value = '';
        contextInput.value = '';
        questionInput.value = '';
        goldInput.value = '';
        goldAnswerGroup.style.display = 'none';
        
        emptyState.style.display = 'flex';
        resultsContent.style.display = 'none';
        currentResults = null;
    });
});
