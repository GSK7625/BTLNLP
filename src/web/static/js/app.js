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
    
    const btnSubmit = document.getElementById('btn-submit');
    const btnClear = document.getElementById('btn-clear');
    
    const resultsLoader = document.getElementById('results-loader');
    const emptyState = document.getElementById('empty-state');
    const resultsContent = document.getElementById('results-content');
    
    // Result cards
    const cardBm25 = document.getElementById('card-bm25');
    const cardPretrained = document.getElementById('card-pretrained');
    const cardFinetuned = document.getElementById('card-finetuned');
    
    // Mode toggling
    const modeReaderOnly = document.getElementById('mode-reader-only');
    const modeRetrieverReader = document.getElementById('mode-retriever-reader');
    const contextGroup = document.getElementById('context-group');
    const retrieverNotice = document.getElementById('retriever-notice');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const highlightViewer = document.getElementById('highlight-viewer');
    
    // Retrieved Contexts DOM elements
    const retrievedContextsSection = document.getElementById('retrieved-contexts-section');
    const retrievedContextsList = document.getElementById('retrieved-contexts-list');
    
    // State variables
    let preloadedExamples = [];
    let currentResults = null;
    let activeTabModel = 'finetuned'; // Default active tab
    let currentMode = 'reader'; // 'reader' or 'pipeline'
    let currentContext = '';
    let currentRetrievedContexts = [];

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
            opt.textContent = qShort;
            exampleSelect.appendChild(opt);
        });
    }

    // Handle Mode toggling
    modeReaderOnly.addEventListener('click', () => {
        currentMode = 'reader';
        modeReaderOnly.classList.add('active');
        modeRetrieverReader.classList.remove('active');
        contextGroup.style.display = 'block';
        if (retrieverNotice) retrieverNotice.style.display = 'none';
        
        // Reset state
        emptyState.style.display = 'flex';
        resultsContent.style.display = 'none';
        currentResults = null;
        currentContext = '';
        currentRetrievedContexts = [];
        retrievedContextsSection.style.display = 'none';
    });

    modeRetrieverReader.addEventListener('click', () => {
        currentMode = 'pipeline';
        modeRetrieverReader.classList.add('active');
        modeReaderOnly.classList.remove('active');
        contextGroup.style.display = 'none';
        if (retrieverNotice) retrieverNotice.style.display = 'block';
        
        // Reset state
        emptyState.style.display = 'flex';
        resultsContent.style.display = 'none';
        currentResults = null;
        currentContext = '';
        currentRetrievedContexts = [];
        retrievedContextsSection.style.display = 'none';
    });

    // Handle example selection
    exampleSelect.addEventListener('change', (e) => {
        const selectedId = e.target.value;
        const descDiv = document.getElementById('example-case-desc');
        
        if (!selectedId) {
            // User chose custom input option
            contextInput.value = '';
            questionInput.value = '';
            goldInput.value = '';
            goldAnswerGroup.style.display = 'none';
            descDiv.style.display = 'none';
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
            
            if (example.case_description) {
                descDiv.textContent = example.case_description;
                descDiv.style.display = 'block';
            } else {
                descDiv.style.display = 'none';
            }
        } else {
            descDiv.style.display = 'none';
        }
    });

    // Handle submit/run
    btnSubmit.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        const context = contextInput.value.trim();
        const gold = goldInput.value.trim();
        
        if (!question) {
            alert('Vui lòng nhập câu hỏi.');
            return;
        }
        
        if (currentMode === 'reader' && !context) {
            alert('Vui lòng nhập đoạn ngữ cảnh.');
            return;
        }

        const models = [];
        if (checkBm25.checked) models.push('bm25');
        if (checkPretrained.checked) models.push('pretrained');
        if (checkFinetuned.checked) models.push('finetuned');
        
        if (models.length === 0) {
            alert('Vui lòng chọn ít nhất một phương pháp để chạy.');
            return;
        }

        // Show loading state
        resultsLoader.style.display = 'flex';
        btnSubmit.disabled = true;
        
        try {
            let response, data;
            if (currentMode === 'reader') {
                response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, context, models, gold })
                });
                data = await response.json();
                currentContext = context;
            } else {
                response = await fetch('/api/predict_pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, models, gold })
                });
                data = await response.json();
                currentContext = data.retrieved_context;
            }
            
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
            
            // Add custom log about retrieval if in pipeline mode
            if (currentMode === 'pipeline') {
                const badge = document.createElement('div');
                badge.style.width = '100%';
                badge.style.padding = '8px 12px';
                badge.style.background = 'rgba(6, 182, 212, 0.15)';
                badge.style.border = '1px solid var(--accent-color)';
                badge.style.borderRadius = 'var(--radius-sm)';
                badge.style.color = '#22d3ee';
                badge.style.fontSize = '12px';
                badge.style.marginBottom = '12px';
                badge.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> BM25 đã truy hồi đoạn văn này từ kho tài liệu mẫu.`;
                
                // Remove existing notice badges in section
                const existing = highlightViewer.parentNode.querySelector('.retrieval-badge');
                if (existing) existing.remove();
                
                badge.className = 'retrieval-badge';
                highlightViewer.parentNode.insertBefore(badge, highlightViewer);
            } else {
                const existing = highlightViewer.parentNode.querySelector('.retrieval-badge');
                if (existing) existing.remove();
            }

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
            const contextToRender = currentResults[activeTabModel] && currentResults[activeTabModel].selected_context 
                ? currentResults[activeTabModel].selected_context 
                : currentContext;
            renderHighlight(contextToRender, activeTabModel);
            
            // Render Top-5 retrieved contexts list
            if (currentMode === 'pipeline') {
                retrievedContextsSection.style.display = 'block';
                currentRetrievedContexts = data.retrieved_contexts || [];
                renderRetrievedContextsList(currentRetrievedContexts, activeTabModel);
            } else {
                retrievedContextsSection.style.display = 'none';
                currentRetrievedContexts = [];
            }
            
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
        
        // Populate BM25
        if (selectedModels.includes('bm25') && results.bm25) {
            const res = results.bm25;
            document.getElementById('ans-bm25').textContent = res.answer || '(Trống)';
            updateGoldBadges('bm25', res);
        }
        
        // Populate Pretrained
        if (selectedModels.includes('pretrained') && results.pretrained) {
            const res = results.pretrained;
            document.getElementById('ans-pretrained').textContent = res.answer || '(Trống)';
            updateGoldBadges('pretrained', res);
        }
        
        // Populate Fine-tuned
        if (selectedModels.includes('finetuned') && results.finetuned) {
            const res = results.finetuned;
            document.getElementById('ans-finetuned').textContent = res.answer || '(Trống)';
            updateGoldBadges('finetuned', res);
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
        if (!context) {
            highlightViewer.textContent = '';
            return;
        }
        
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
            const contextToRender = currentResults[activeTabModel] && currentResults[activeTabModel].selected_context 
                ? currentResults[activeTabModel].selected_context 
                : currentContext;
            renderHighlight(contextToRender, activeTabModel);
            
            // Rerender retrieved contexts list to update active selection badge
            if (currentMode === 'pipeline') {
                renderRetrievedContextsList(currentRetrievedContexts, activeTabModel);
            }
        });
    });

    // Render list of top-5 retrieved contexts
    function renderRetrievedContextsList(contexts, activeModel) {
        retrievedContextsList.innerHTML = '';
        if (!contexts || contexts.length === 0) return;
        
        const selectedContext = currentResults[activeModel] && currentResults[activeModel].selected_context
            ? currentResults[activeModel].selected_context
            : null;
            
        contexts.forEach((ctx, index) => {
            const div = document.createElement('div');
            div.className = 'retrieved-item';
            
            const isActive = (selectedContext === ctx);
            if (isActive) {
                div.classList.add('active-selection');
            }
            
            // Model label mapping for display badge
            const modelLabelMap = {
                'bm25': 'B1: BM25 Sentence',
                'pretrained': 'B2: Pretrained XLM-R',
                'finetuned': 'M1: Fine-tuned XLM-R'
            };
            
            const badgeHtml = isActive 
                ? `<span class="retrieved-badge-selected"><i class="fa-solid fa-circle-check"></i> Đang chọn bởi ${modelLabelMap[activeModel] || activeModel}</span>`
                : '';
                
            div.innerHTML = `
                <div class="retrieved-item-header">
                    <span>Đoạn #${index + 1} (BM25 Rank ${index + 1})</span>
                    ${badgeHtml}
                </div>
                <div class="retrieved-item-body">
                    ${escapeHtml(ctx)}
                </div>
            `;
            retrievedContextsList.appendChild(div);
        });
    }

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
        currentContext = '';
        currentRetrievedContexts = [];
        retrievedContextsSection.style.display = 'none';
        retrievedContextsList.innerHTML = '';
        
        const existing = highlightViewer.parentNode.querySelector('.retrieval-badge');
        if (existing) existing.remove();
    });

});
