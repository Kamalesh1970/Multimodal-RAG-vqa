/**
 * Centralized API client module with Firebase authentication integration.
 * Uses relative paths to communicate with the co-hosted FastAPI backend.
 */
const API = {
    // API Configurations
    TIMEOUT_MS: 120000, // 2 minutes timeout for document uploads / models
    idToken: null,      // Active Firebase ID Token

    /**
     * Stores the current Firebase ID Token.
     */
    setToken(token) {
        API.idToken = token;
    },

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

    async fetchWithAuth(resource, options = {}) {
        const headers = { ... (options.headers || {}) };
        
        let firebaseUser = null;
        try {
            if (window.firebase && firebase.auth && firebase.auth().currentUser) {
                firebaseUser = firebase.auth().currentUser;
            }
        } catch (_) {}
        
        if (firebaseUser) {
            try {
                // Dynamically fetch the current valid ID token
                const token = await firebaseUser.getIdToken();
                headers['Authorization'] = `Bearer ${token}`;
            } catch (err) {
                console.error("Failed to dynamically acquire Firebase ID token:", err);
            }
        } else if (API.idToken) {
            // Fallback for static mock token usage
            headers['Authorization'] = `Bearer ${API.idToken}`;
        }
        
        let response = await API.fetchWithTimeout(resource, {
            ...options,
            headers
        });
        
        // Single retry on 401 Unauthorized using forced token refresh
        if (response.status === 401 && firebaseUser) {
            console.warn("Received 401. Forcing Firebase ID token refresh and retrying...");
            try {
                const refreshedToken = await firebaseUser.getIdToken(true);
                headers['Authorization'] = `Bearer ${refreshedToken}`;
                response = await API.fetchWithTimeout(resource, {
                    ...options,
                    headers
                });
                
                if (response.status === 401) {
                    console.error("Token refresh retry failed with 401. Signing out user.");
                    await firebase.auth().signOut();
                }
            } catch (refreshErr) {
                console.error("Error refreshing token after 401:", refreshErr);
                try {
                    await firebase.auth().signOut();
                } catch (_) {}
            }
        }
        
        return response;
    },

    /**
     * Retrieves public Firebase Client SDK Configuration.
     * GET /auth/config
     */
    async getFirebaseConfig() {
        try {
            const response = await API.fetchWithTimeout('/auth/config', { timeout: 5000 });
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            return await response.json();
        } catch (err) {
            console.error('Failed to load Firebase client config:', err);
            throw err;
        }
    },

    /**
     * Checks connection and queries system status.
     * GET /system/status
     */
    async checkHealth() {
        try {
            const response = await API.fetchWithAuth('/system/status', { timeout: 5000 });
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
     * Syncs user profile with the backend database.
     * POST /auth/sync
     */
    async syncUserProfile(displayName) {
        try {
            const response = await API.fetchWithAuth('/auth/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ display_name: displayName })
            });
            if (!response.ok) {
                throw new Error('Failed to sync profile.');
            }
            return await response.json();
        } catch (err) {
            console.error('User profile sync failed:', err);
            throw err;
        }
    },

    /**
     * Lists metadata for all documents owned by the active user.
     * GET /documents
     */
    async listDocuments() {
        try {
            const response = await API.fetchWithAuth('/documents');
            if (!response.ok) {
                throw new Error('Could not retrieve document list.');
            }
            return await response.json();
        } catch (err) {
            console.error('Fetch documents list failed:', err);
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
            const response = await API.fetchWithAuth('/documents/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errMsg = 'Upload failed.';
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
            const response = await API.fetchWithAuth(`/documents/${docId}`);
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
     * Deletes a document owned by the active user.
     * DELETE /documents/{doc_id}
     */
    async deleteDocument(docId) {
        try {
            const response = await API.fetchWithAuth(`/documents/${docId}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                const error = new Error('Could not delete document.');
                error.status = response.status;
                throw error;
            }
            return await response.json();
        } catch (err) {
            console.error('Delete document failed:', err);
            throw err;
        }
    },

    /**
     * Ask a question about an active document.
     * POST /ask
     */
    async askQuestion(docId, question) {
        try {
            const response = await API.fetchWithAuth('/ask', {
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
