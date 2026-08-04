/**
 * Main application controller managing UI state, Firebase authentication,
 * document selector dashboard, and API communication.
 */
document.addEventListener('DOMContentLoaded', async () => {
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
    const userDisplay = document.getElementById('user-display');
    const logoutBtn = document.getElementById('logout-btn');
    
    // Auth Panel
    const authLoading = document.getElementById('auth-loading');
    const authContainer = document.getElementById('auth-container');
    const authCard = document.querySelector('.auth-card');
    const authTitle = document.getElementById('auth-title');
    const authSubtitle = document.getElementById('auth-subtitle');
    const authForm = document.getElementById('auth-form');
    const authEmail = document.getElementById('auth-email');
    const authPassword = document.getElementById('auth-password');
    const authName = document.getElementById('auth-name');
    const authConfirmPassword = document.getElementById('auth-confirm-password');
    const nameGroup = document.getElementById('name-group');
    const confirmPasswordGroup = document.getElementById('confirm-password-group');
    const authSubmitBtn = document.getElementById('auth-submit-btn');
    const authSwitchLink = document.getElementById('auth-switch-link');
    const switchText = document.getElementById('switch-text');
    const authError = document.getElementById('auth-error');

    // Layout
    const appLayout = document.querySelector('.app-layout');
    const appHeader = document.querySelector('.app-header');
    const appFooter = document.querySelector('.app-footer-bar');
    
    // Panel Headers & Warnings
    const newDocBtn = document.getElementById('new-doc-btn');
    const simulatedWarning = document.getElementById('simulated-warning');
    
    // Document Selector Dashboard
    const docSelectorContainer = document.getElementById('doc-selector-container');
    const docSelect = document.getElementById('doc-select');
    const deleteDocBtn = document.getElementById('delete-doc-btn');

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

    // --- Firebase Client SDK Initialization ---
    let authMode = 'login'; // 'login' or 'register'

    try {
        const config = await API.getFirebaseConfig();
        // Bypassing login card overlay and loading the main dashboard directly in single-user mode.
        API.setToken(null);
        
        userDisplay.textContent = config.firebase_enabled !== false ? "Firebase Online Mode" : "Local SQLite Mode";
        userDisplay.style.display = 'inline-block';
        logoutBtn.style.display = 'none'; // Hide logout button in single-user mode
        
        if (authLoading) authLoading.style.display = 'none';
        authContainer.style.display = 'none';
        appLayout.style.display = 'flex';
        appHeader.style.display = 'flex';
        appFooter.style.display = 'block';
        
        checkSystemConnection();
        await loadUserDocuments();
    } catch (err) {
        showGlobalError("Failed to initialize system interface.");
        if (authLoading) authLoading.style.display = 'none';
    }

    function setupAuthObserver() {
        firebase.auth().onIdTokenChanged(async (user) => {
            if (user) {
                // User logged in / token refreshed
                try {
                    const token = await user.getIdToken();
                    API.setToken(token);
                    
                    userDisplay.textContent = `Welcome, ${user.displayName || user.email}`;
                    userDisplay.style.display = 'inline-block';
                    logoutBtn.style.display = 'inline-block';
                    
                    if (authLoading) authLoading.style.display = 'none';
                    authContainer.style.display = 'none';
                    appLayout.style.display = 'flex';
                    appHeader.style.display = 'flex';
                    appFooter.style.display = 'block';
                    
                    // Trigger connection checks
                    checkSystemConnection();
                    
                    // Load user document library
                    await loadUserDocuments();
                } catch (err) {
                    console.error("Token acquisition failed:", err);
                }
            } else {
                // User logged out
                API.setToken(null);
                userDisplay.style.display = 'none';
                logoutBtn.style.display = 'none';
                appLayout.style.display = 'none';
                appHeader.style.display = 'none';
                appFooter.style.display = 'none';
                if (authLoading) authLoading.style.display = 'none';
                authContainer.style.display = 'flex';
                resetSession();
            }
        });
    }

    // --- Authentication UI Handlers ---
    authSwitchLink.addEventListener('click', (e) => {
        e.preventDefault();
        authError.style.display = 'none';
        
        if (authMode === 'login') {
            authMode = 'register';
            authTitle.textContent = 'Register';
            authSubtitle.textContent = 'Create a secure account to analyze your documents.';
            nameGroup.style.display = 'flex';
            confirmPasswordGroup.style.display = 'flex';
            authSubmitBtn.textContent = 'Register';
            switchText.textContent = 'Already have an account?';
            authSwitchLink.textContent = 'Sign In';
            authConfirmPassword.required = true;
        } else {
            authMode = 'login';
            authTitle.textContent = 'Sign In';
            authSubtitle.textContent = 'Enter your email and password to access the app.';
            nameGroup.style.display = 'none';
            confirmPasswordGroup.style.display = 'none';
            authSubmitBtn.textContent = 'Sign In';
            switchText.textContent = "Don't have an account?";
            authSwitchLink.textContent = 'Register';
            authConfirmPassword.required = false;
        }
    });

    authForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        authError.style.display = 'none';
        
        const email = authEmail.value.trim();
        const password = authPassword.value;
        
        if (authMode === 'register') {
            const name = authName.value.trim();
            const confirmPass = authConfirmPassword.value;
            
            if (password !== confirmPass) {
                authError.textContent = "Passwords do not match.";
                authError.style.display = 'block';
                return;
            }
            
            authSubmitBtn.disabled = true;
            authSubmitBtn.textContent = 'Registering...';
            try {
                const creds = await firebase.auth().createUserWithEmailAndPassword(email, password);
                if (name) {
                    await creds.user.updateProfile({ displayName: name });
                }
                
                // Get token and sync user profile to database
                const token = await creds.user.getIdToken();
                API.setToken(token);
                await API.syncUserProfile(name || email);
                
                authSubmitBtn.disabled = false;
                authSubmitBtn.textContent = 'Register';
            } catch (err) {
                authError.textContent = err.message;
                authError.style.display = 'block';
                authSubmitBtn.disabled = false;
                authSubmitBtn.textContent = 'Register';
            }
        } else {
            authSubmitBtn.disabled = true;
            authSubmitBtn.textContent = 'Signing In...';
            try {
                await firebase.auth().signInWithEmailAndPassword(email, password);
                authSubmitBtn.disabled = false;
                authSubmitBtn.textContent = 'Sign In';
            } catch (err) {
                authError.textContent = err.message;
                authError.style.display = 'block';
                authSubmitBtn.disabled = false;
                authSubmitBtn.textContent = 'Sign In';
            }
        }
    });

    logoutBtn.addEventListener('click', async () => {
        try {
            await firebase.auth().signOut();
        } catch (err) {
            console.error('Logout failed:', err);
        }
    });

    // --- Document Selector Dashboard Logic ---
    async function loadUserDocuments(selectDocId = null) {
        try {
            const documents = await API.listDocuments();
            
            docSelect.innerHTML = '';
            
            if (documents.length === 0) {
                docSelectorContainer.style.display = 'none';
                resetSession();
                return;
            }
            
            documents.forEach(doc => {
                const opt = document.createElement('option');
                opt.value = doc.doc_id;
                opt.textContent = `${doc.filename} (${doc.status})`;
                docSelect.appendChild(opt);
            });
            
            docSelectorContainer.style.display = 'flex';
            
            const targetId = selectDocId || documents[0].doc_id;
            docSelect.value = targetId;
            
            await selectActiveDocument(targetId);
        } catch (err) {
            console.error('Failed to load user documents:', err);
            showGlobalError('Failed to load document list.');
        }
    }

    async function selectActiveDocument(docId) {
        try {
            activeDocId = docId;
            const docInfo = await API.getDocumentMetadata(docId);
            renderDocumentDetails(docInfo);
            
            if (docInfo.status === 'completed') {
                questionInput.disabled = false;
                questionInput.placeholder = "Ask a question about this document...";
                sendBtn.disabled = false;
                newDocBtn.style.display = 'inline-block';
                
                // Clear chat panel on switch
                conversationContainer.innerHTML = '';
                conversationContainer.style.display = 'none';
                emptyQaState.style.display = 'flex';
            } else {
                questionInput.disabled = true;
                questionInput.placeholder = `Document status is: ${docInfo.status}...`;
                sendBtn.disabled = true;
                newDocBtn.style.display = 'none';
            }
        } catch (err) {
            console.error('Failed to select active document:', err);
        }
    }

    docSelect.addEventListener('change', (e) => {
        selectActiveDocument(e.target.value);
    });

    deleteDocBtn.addEventListener('click', async () => {
        if (!activeDocId) return;
        if (!confirm('Are you sure you want to delete this document and all its data permanently?')) return;
        
        try {
            await API.deleteDocument(activeDocId);
            showGlobalError('Document deleted successfully.');
            await loadUserDocuments();
        } catch (err) {
            showGlobalError('Failed to delete document: ' + err.message);
        }
    });

    // --- Helper Functions ---
    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    async function checkSystemConnection() {
        let firebaseEnabled = false;
        try {
            const config = await API.getFirebaseConfig();
            firebaseEnabled = config.firebase_enabled !== false;
        } catch (_) {}
        
        if (firebaseEnabled && (!window.firebase || !firebase.auth || !firebase.auth().currentUser)) {
            return;
        }
        
        try {
            const data = await API.checkHealth();
            isConnected = true;
            connectionBadge.className = 'badge badge-connected';
            
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

    function showGlobalError(msg) {
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
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

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

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        if (isUploading) return;
        
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        const supported = ['.pdf', '.png', '.jpg', '.jpeg'];
        
        if (!supported.includes(ext)) {
            showGlobalError("This file type isn't supported. Upload a JPG, PNG or PDF.");
            return;
        }

        if (file.size > 20 * 1024 * 1024) {
            showGlobalError("File is too large. Maximum size is 20MB.");
            return;
        }

        uploadFile(file);
    }

    async function uploadFile(file) {
        isUploading = true;
        dropzone.style.display = 'none';
        processingContainer.style.display = 'flex';
        
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

            clearInterval(processingInterval);
            processingContainer.style.display = 'none';
            
            // Reload user documents to sync dashboard selector and display details
            await loadUserDocuments(activeDocId);
            
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

        docActiveView.style.display = 'flex';
        dropzone.style.display = 'none';
        updatePreview();
    }

    function updatePreview() {
        previewViewport.innerHTML = '';
        
        if (pageCount > 1) {
            previewPager.style.display = 'flex';
            currentPageNum.textContent = currentPage;
            totalPagesNum.textContent = pageCount;
            
            prevPageBtn.disabled = currentPage === 1;
            nextPageBtn.disabled = currentPage === pageCount;
        } else {
            previewPager.style.display = 'none';
        }

        const imgUrl = `/processed/${activeDocId}/page_${currentPage}.jpg`;
        
        // Fetch preprocessed preview image securely using authorization token
        API.fetchWithAuth(imgUrl)
            .then(res => {
                if (!res.ok) {
                    throw new Error("Unauthorized preview access");
                }
                return res.blob();
            })
            .then(blob => {
                const objectUrl = URL.createObjectURL(blob);
                const img = document.createElement('img');
                img.className = 'preview-image';
                img.src = objectUrl;
                img.alt = `Document Page ${currentPage}`;
                img.onload = () => {
                    URL.revokeObjectURL(objectUrl);
                };
                img.onerror = () => {
                    previewViewport.innerHTML = `<div class="preview-empty">Preview not generated for Page ${currentPage}</div>`;
                };
                previewViewport.appendChild(img);
            })
            .catch(err => {
                previewViewport.innerHTML = `<div class="preview-empty">Access Denied or Preview Unavailable</div>`;
            });
    }

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

    newDocBtn.addEventListener('click', () => {
        resetSession();
    });

    function resetSession() {
        if (isUploading || isAnswering) return;

        activeDocId = null;
        pageCount = 0;
        currentPage = 1;
        fileFormat = null;
        
        if (processingInterval) {
            clearInterval(processingInterval);
        }

        docActiveView.style.display = 'none';
        previewViewport.innerHTML = '<div class="preview-empty">No preview available</div>';
        previewPager.style.display = 'none';
        newDocBtn.style.display = 'none';
        
        questionInput.value = '';
        questionInput.disabled = true;
        questionInput.placeholder = "Upload a document to begin asking...";
        sendBtn.disabled = true;
        
        conversationContainer.innerHTML = '';
        conversationContainer.style.display = 'none';
        emptyQaState.style.display = 'flex';
        
        dropzone.style.display = 'flex';
        fileInput.value = '';
    }

    document.querySelectorAll('.btn-suggestion').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-q');
            if (query && !questionInput.disabled && !isAnswering) {
                questionInput.value = query;
                submitQuestion();
            }
        });
    });

    sendBtn.addEventListener('click', () => {
        submitQuestion();
    });

    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuestion();
        }
    });

    async function submitQuestion() {
        const question = questionInput.value.trim();
        if (!question || !activeDocId || isAnswering || !isConnected) return;

        isAnswering = true;
        questionInput.disabled = true;
        sendBtn.disabled = true;
        
        emptyQaState.style.display = 'none';
        conversationContainer.style.display = 'flex';
        answerLoader.style.display = 'flex';
        
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

        appendUserMessage(question);

        try {
            const askResponse = await API.askQuestion(activeDocId, question);
            
            clearInterval(statusTimer);
            answerLoader.style.display = 'none';
            
            appendSystemResponse(askResponse);
            
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

    function appendSystemResponse(data) {
        const lastTurn = conversationContainer.lastElementChild;
        if (!lastTurn) return;

        const systemDiv = document.createElement('div');
        systemDiv.className = 'chat-system-answer';

        const ansBody = document.createElement('div');
        ansBody.className = 'answer-body';

        if (!data.answerable) {
            ansBody.textContent = "I couldn't find enough evidence in this document to answer that question. Try asking about information visible in the uploaded document.";
            systemDiv.appendChild(ansBody);
        } else {
            ansBody.textContent = data.answer;
            systemDiv.appendChild(ansBody);

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

        const details = document.createElement('details');
        details.className = 'answer-details';
        
        const summary = document.createElement('summary');
        summary.textContent = 'Grounding Details';
        details.appendChild(summary);

        const detailsContent = document.createElement('div');
        detailsContent.className = 'answer-details-content';
        
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
});
