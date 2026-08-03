/**
 * Centralized API client module.
 * Uses relative paths to communicate with the co-hosted FastAPI backend.
 */
const API = {
    // API Configurations
    TIMEOUT_MS: 120000, // 2 minutes timeout for document uploads / models

    /**
     * Executes a fetch request with a timeout.
     */
    async fetchWithTimeout(resource, options = {}) {
        const { timeout = API.TIMEOUT_MS } = options;
        
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeout);
        
        const response = await fetch(resource, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    },

    /**
     * Checks connection and queries system status.
     * GET /system/status
     */
    async checkHealth() {
        try {
            const response = await API.fetchWithTimeout('/system/status', { timeout: 5000 });
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            return await response.json();
        } catch (err) {
            console.error('System health check failed:', err);
            throw err;
        }
    },

    /**
     * Uploads document to database.
     * POST /documents/upload
     */
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Uploading may take longer (OCR, embedding calculations), use full timeout
            const response = await API.fetchWithTimeout('/documents/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errMsg = 'Upload failed.';
                try {
                    const errDetail = await response.json();
                    errMsg = errDetail.detail || errMsg;
                } catch (_) {}
                
                // Map known statuses
                const error = new Error(errMsg);
                error.status = response.status;
                throw error;
            }

            return await response.json();
        } catch (err) {
            console.error('Document upload error:', err);
            throw err;
        }
    },

    /**
     * Retrieves metadata for an active document.
     * GET /documents/{doc_id}
     */
    async getDocumentMetadata(docId) {
        try {
            const response = await API.fetchWithTimeout(`/documents/${docId}`);
            if (!response.ok) {
                const error = new Error('Could not retrieve metadata.');
                error.status = response.status;
                throw error;
            }
            return await response.json();
        } catch (err) {
            console.error('Fetch document metadata failed:', err);
            throw err;
        }
    },

    /**
     * Ask a question about an active document.
     * POST /ask
     */
    async askQuestion(docId, question) {
        try {
            const response = await API.fetchWithTimeout('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    doc_id: docId,
                    question: question
                })
            });

            if (!response.ok) {
                let errMsg = 'Question generation failed.';
                try {
                    const errDetail = await response.json();
                    errMsg = errDetail.detail || errMsg;
                } catch (_) {}
                
                const error = new Error(errMsg);
                error.status = response.status;
                throw error;
            }

            return await response.json();
        } catch (err) {
            console.error('Question execution failed:', err);
            throw err;
        }
    }
};
