/**
 * Main application controller managing UI state, event handling,
 * and relative path API communication.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- Application State ---
    let activeDocId = null;
    let pageCount = 0;
    let currentPage = 1;
    let isUploading = false;
    let isAnswering = false;
    let fileFormat = null;
    let isConnected = false;
    let processingInterval = null;

    // --- DOM Elements ---
    // Header
    const connectionBadge = document.getElementById('connection-badge');
    
    // Panel Headers & Warnings
    const newDocBtn = document.getElementById('new-doc-btn');
    const simulatedWarning = document.getElementById('simulated-warning');
    
    // Upload & Ingestion
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const processingContainer = document.getElementById('processing-container');
    const processingMessage = document.getElementById('processing-message');
    
    // Document View & Metadata
    const docActiveView = document.getElementById('doc-active-view');
    const metaFilename = document.getElementById('meta-filename');
    const metaFormat = document.getElementById('meta-format');
    const metaPagesRow = document.getElementById('meta-pages-row');
    const metaPages = document.getElementById('meta-pages');
    const metaDocId = document.getElementById('meta-doc-id');
    
    // Preview
    const previewViewport = document.getElementById('preview-viewport');
    const previewPager = document.getElementById('preview-pager');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const currentPageNum = document.getElementById('current-page-num');
    const totalPagesNum = document.getElementById('total-pages-num');
    
    // Q&A Area
    const emptyQaState = document.getElementById('empty-qa-state');
    const conversationContainer = document.getElementById('conversation-container');
    const answerLoader = document.getElementById('answer-loader');
    const loaderStatus = document.getElementById('loader-status');
    const questionInput = document.getElementById('question-input');
    const sendBtn = document.getElementById('send-btn');

    // --- Helper Functions ---

    /**
     * Escape string content safely for HTML interpolation.
     */
    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    /**
     * Updates the connection badge status.
     */
    async function checkSystemConnection() {
        try {
            const data = await API.checkHealth();
            isConnected = true;
            connectionBadge.className = 'badge badge-connected';
            
            // Map generation mode
            const modeLabel = data.generation_mode === 'simulated' ? 'Simulated' : 'Live';
            connectionBadge.textContent = `Connected (${modeLabel})`;
            
            if (data.generation_mode === 'simulated') {
                simulatedWarning.style.display = 'inline-block';
            } else {
                simulatedWarning.style.display = 'none';
            }
        } catch (err) {
            isConnected = false;
            connectionBadge.className = 'badge badge-disconnected';
            connectionBadge.textContent = 'Backend Offline';
            simulatedWarning.style.display = 'none';
            showGlobalError('Backend connection lost. Unable to query documents at this time.');
        }
    }

    /**
     * Displays a global error message box.
     */
    function showGlobalError(msg) {
        // Find or create global toast
        let toast = document.getElementById('error-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'error-toast';
            toast.style.position = 'fixed';
            toast.style.bottom = '16px';
            toast.style.right = '16px';
            toast.style.backgroundColor = '#ffebee';
            toast.style.color = '#c62828';
            toast.style.border = '1px solid #ffcdd2';
            toast.style.padding = '12px 16px';
            toast.style.borderRadius = '4px';
            toast.style.fontSize = '13px';
            toast.style.fontWeight = '500';
            toast.style.zIndex = '1000';
            toast.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
            document.body.appendChild(toast);
        }
        toast.textContent = msg;
        toast.style.display = 'block';
        setTimeout(() => {
            toast.style.display = 'none';
        }, 6000);
    }

    // --- Drag and Drop Logic ---

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.add('dropzone-dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.remove('dropzone-dragover');
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // Handle click browser upload
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    /**
     * Validates and prepares file upload.
     */
    function handleFileSelection(file) {
        if (isUploading) return;
        
        // 1. Validation
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        const supported = ['.pdf', '.png', '.jpg', '.jpeg'];
        
        if (!supported.includes(ext)) {
            showGlobalError("This file type isn't supported. Upload a JPG, PNG or PDF.");
            return;
        }

        // 20MB limit
        if (file.size > 20 * 1024 * 1024) {
            showGlobalError("File is too large. Maximum size is 20MB.");
            return;
        }

        uploadFile(file);
    }

    /**
     * Executes upload and updates state steps.
     */
    async function uploadFile(file) {
        isUploading = true;
        dropzone.style.display = 'none';
        processingContainer.style.display = 'flex';
        
        // Rotate status messages to represent actual pipeline steps
        const stages = [
            "Uploading document...",
            "Processing page layouts...",
            "Extracting document text (OCR)...",
            "Generating multimodal embeddings...",
            "Indexing database records..."
        ];
        let stageIdx = 0;
        processingMessage.textContent = stages[0];
        
        processingInterval = setInterval(() => {
            if (stageIdx < stages.length - 1) {
                stageIdx++;
                processingMessage.textContent = stages[stageIdx];
            }
        }, 2200);

        try {
            const uploadResult = await API.uploadDocument(file);
            activeDocId = uploadResult.doc_id;

            // Fetch document metadata details
            const docInfo = await API.getDocumentMetadata(activeDocId);
            
            clearInterval(processingInterval);
            processingContainer.style.display = 'none';
            
            // Render active document panel
            renderDocumentDetails(docInfo);
            
            // Unlock Q&A controls
            questionInput.disabled = false;
            questionInput.placeholder = "Ask a question about this document...";
            sendBtn.disabled = false;
            newDocBtn.style.display = 'inline-block';
            questionInput.focus();
            
            isUploading = false;
        } catch (err) {
            clearInterval(processingInterval);
            processingContainer.style.display = 'none';
            dropzone.style.display = 'flex';
            isUploading = false;
            
            let message = "Upload failed. The server could not process this document.";
            if (err.status === 413) {
                message = "The uploaded file exceeds the server payload size limit.";
            } else if (err.message) {
                message = `Upload error: ${err.message}`;
            }
            showGlobalError(message);
        }
    }

    /**
     * Renders loaded document details and image preview.
     */
    function renderDocumentDetails(docInfo) {
        metaFilename.textContent = docInfo.filename;
        metaFilename.title = docInfo.filename;
        metaFormat.textContent = docInfo.file_type.toUpperCase();
        metaDocId.textContent = docInfo.doc_id;

        pageCount = docInfo.page_count;
        fileFormat = docInfo.file_type.toLowerCase();
        currentPage = 1;

        if (pageCount > 0) {
            metaPagesRow.style.display = 'flex';
            metaPages.textContent = `${pageCount} page${pageCount > 1 ? 's' : ''}`;
        } else {
            metaPagesRow.style.display = 'none';
        }

        // Show metadata container
        docActiveView.style.display = 'flex';

        // Render preview image
        updatePreview();
    }

    /**
     * Renders the current page preview.
     */
    function updatePreview() {
        previewViewport.innerHTML = '';
        
        if (pageCount > 1) {
            // PDF: multiple pages preview
            previewPager.style.display = 'flex';
            currentPageNum.textContent = currentPage;
            totalPagesNum.textContent = pageCount;
            
            prevPageBtn.disabled = currentPage === 1;
            nextPageBtn.disabled = currentPage === pageCount;
        } else {
            previewPager.style.display = 'none';
        }

        // Generate static image url
        // Relies on static mount /processed/{doc_id}/page_{page_num}.jpg
        const imgUrl = `/processed/${activeDocId}/page_${currentPage}.jpg`;
        
        const img = document.createElement('img');
        img.className = 'preview-image';
        img.src = imgUrl;
        img.alt = `Document Page ${currentPage}`;
        img.onerror = () => {
            previewViewport.innerHTML = `<div class="preview-empty">Preview not generated for Page ${currentPage}</div>`;
        };
        
        previewViewport.appendChild(img);
    }

    // Pager controls
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            updatePreview();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        if (currentPage < pageCount) {
            currentPage++;
            updatePreview();
        }
    });

    // --- Reset Document Session ---
    newDocBtn.addEventListener('click', () => {
        resetSession();
    });

    function resetSession() {
        if (isUploading || isAnswering) return;

        // Clear active vars
        activeDocId = null;
        pageCount = 0;
        currentPage = 1;
        fileFormat = null;
        
        if (processingInterval) {
            clearInterval(processingInterval);
        }

        // Reset UI Panels
        docActiveView.style.display = 'none';
        previewViewport.innerHTML = '<div class="preview-empty">No preview available</div>';
        previewPager.style.display = 'none';
        newDocBtn.style.display = 'none';
        
        // Reset Inputs
        questionInput.value = '';
        questionInput.disabled = true;
        questionInput.placeholder = "Upload a document to begin asking...";
        sendBtn.disabled = true;
        
        // Reset Q&A history logs
        conversationContainer.innerHTML = '';
        conversationContainer.style.display = 'none';
        emptyQaState.style.display = 'flex';
        
        // Show upload dropzone
        dropzone.style.display = 'flex';
        fileInput.value = '';
    }

    // --- Suggestion Questions Click handler ---
    document.querySelectorAll('.btn-suggestion').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-q');
            if (query && !questionInput.disabled && !isAnswering) {
                questionInput.value = query;
                submitQuestion();
            }
        });
    });

    // --- Q&A Submit Logic ---

    sendBtn.addEventListener('click', () => {
        submitQuestion();
    });

    questionInput.addEventListener('keydown', (e) => {
        // Send on Enter, newline on Shift+Enter
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuestion();
        }
    });

    /**
     * Submit question to endpoint.
     */
    async function submitQuestion() {
        const question = questionInput.value.trim();
        if (!question || !activeDocId || isAnswering || !isConnected) return;

        isAnswering = true;
        questionInput.disabled = true;
        sendBtn.disabled = true;
        
        // Show inline loader
        emptyQaState.style.display = 'none';
        conversationContainer.style.display = 'flex';
        answerLoader.style.display = 'flex';
        
        // Rotate query status indicators
        const searchStages = [
            "Searching document...",
            "Finding relevant evidence...",
            "Generating grounded answer..."
        ];
        let idx = 0;
        loaderStatus.textContent = searchStages[0];
        const statusTimer = setInterval(() => {
            if (idx < searchStages.length - 1) {
                idx++;
                loaderStatus.textContent = searchStages[idx];
            }
        }, 1500);

        // Prepend user turn to chat viewport immediately
        appendUserMessage(question);

        try {
            const askResponse = await API.askQuestion(activeDocId, question);
            
            clearInterval(statusTimer);
            answerLoader.style.display = 'none';
            
            // Render grounded system response
            appendSystemResponse(askResponse);
            
            // Clear input and focus
            questionInput.value = '';
            questionInput.disabled = false;
            sendBtn.disabled = false;
            questionInput.focus();
            
            isAnswering = false;
        } catch (err) {
            clearInterval(statusTimer);
            answerLoader.style.display = 'none';
            
            questionInput.disabled = false;
            sendBtn.disabled = false;
            isAnswering = false;
            
            let message = "Ask failed. The server could not generate an answer.";
            if (err.status === 429) {
                message = "The configured VLM provider quota has been exceeded. Try again later.";
            } else if (err.status === 503 || err.status === 502) {
                message = "The model provider service is temporarily unavailable.";
            } else if (err.message) {
                message = `Error: ${err.message}`;
            }
            
            appendErrorTurn(message);
        }
    }

    /**
     * Appends User Question UI node to conversation container.
     */
    function appendUserMessage(text) {
        const turnDiv = document.createElement('div');
        turnDiv.className = 'chat-turn';
        
        const qDiv = document.createElement('div');
        qDiv.className = 'chat-user-question';
        qDiv.textContent = text;
        
        turnDiv.appendChild(qDiv);
        conversationContainer.appendChild(turnDiv);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }

    /**
     * Appends Grounded System Response UI node to conversation container.
     */
    function appendSystemResponse(data) {
        const lastTurn = conversationContainer.lastElementChild;
        if (!lastTurn) return;

        const systemDiv = document.createElement('div');
        systemDiv.className = 'chat-system-answer';

        // 1. Answer text container
        const ansBody = document.createElement('div');
        ansBody.className = 'answer-body';

        if (!data.answerable) {
            ansBody.textContent = "I couldn't find enough evidence in this document to answer that question. Try asking about information visible in the uploaded document.";
            systemDiv.appendChild(ansBody);
        } else {
            ansBody.textContent = data.answer;
            systemDiv.appendChild(ansBody);

            // 2. Evidence block
            if (data.evidence && data.evidence.length > 0) {
                const evidenceSec = document.createElement('div');
                evidenceSec.className = 'evidence-section';
                
                const header = document.createElement('div');
                header.className = 'evidence-header';
                header.textContent = 'Evidence Cited';
                evidenceSec.appendChild(header);

                data.evidence.forEach(item => {
                    const quote = document.createElement('div');
                    quote.className = 'evidence-quote';
                    quote.textContent = `"${item.text}"`;
                    evidenceSec.appendChild(quote);
                });

                const meta = document.createElement('div');
                meta.className = 'evidence-meta';
                
                const pagesLabel = document.createElement('span');
                const pageList = data.pages_used.join(', ');
                pagesLabel.textContent = `Source: Page${data.pages_used.length > 1 ? 's' : ''} ${pageList}`;
                meta.appendChild(pagesLabel);

                const typeSpan = document.createElement('span');
                typeSpan.className = 'meta-item-badge';
                typeSpan.textContent = data.grounding_type.toUpperCase();
                meta.appendChild(typeSpan);

                evidenceSec.appendChild(meta);
                systemDiv.appendChild(evidenceSec);
            }
        }

        // 3. Technical collapsible details
        const details = document.createElement('details');
        details.className = 'answer-details';
        
        const summary = document.createElement('summary');
        summary.textContent = 'Grounding Details';
        details.appendChild(summary);

        const detailsContent = document.createElement('div');
        detailsContent.className = 'answer-details-content';
        
        // Score & counts
        let retPages = [];
        if (data.retrieval) {
            retPages = data.retrieval.pages_considered || [];
        }

        detailsContent.innerHTML = `
            <div><strong>Grounding Status:</strong> ${escapeHtml(data.grounding_type)}</div>
            <div><strong>Explanation:</strong> ${escapeHtml(data.grounding_explanation)}</div>
            <div><strong>Retrieval Top Score:</strong> ${data.retrieval ? data.retrieval.top_score.toFixed(4) : 'N/A'}</div>
            <div><strong>Pages Evaluated:</strong> [${retPages.join(', ')}]</div>
        `;
        
        details.appendChild(detailsContent);
        systemDiv.appendChild(details);

        lastTurn.appendChild(systemDiv);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }

    /**
     * Appends an error notice block inside conversation log.
     */
    function appendErrorTurn(errMsg) {
        const lastTurn = conversationContainer.lastElementChild;
        if (!lastTurn) return;

        const errorDiv = document.createElement('div');
        errorDiv.style.color = 'var(--error)';
        errorDiv.style.fontSize = '13.5px';
        errorDiv.style.marginTop = '8px';
        errorDiv.style.padding = '8px 12px';
        errorDiv.style.backgroundColor = '#ffebee';
        errorDiv.style.borderLeft = '3px solid var(--error)';
        errorDiv.style.borderRadius = '0 3px 3px 0';
        errorDiv.textContent = errMsg;

        lastTurn.appendChild(errorDiv);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }

    // --- Startup Loop ---
    checkSystemConnection();
    // Refresh health status every 30 seconds
    setInterval(checkSystemConnection, 30000);
});
