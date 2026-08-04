import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force Firebase enabled for this test script
os.environ["FIREBASE_ENABLED"] = "true"

from backend.config import settings
from backend.firebase.client import initialize_firebase, get_firestore_client
from google.cloud import firestore

def main():
    print("Testing Firebase/Cloud Firestore connectivity...")
    
    # 1. Initialize Firebase
    success = initialize_firebase()
    if not success:
        print("FIREBASE CONNECTION: FAIL (Initialization failed)")
        sys.exit(1)
        
    db = get_firestore_client()
    if db is None:
        print("FIREBASE CONNECTION: FAIL (Could not acquire Firestore client)")
        sys.exit(1)
        
    doc_ref = db.collection("system").document("connection_test")
    
    try:
        # 3. Write a temporary document
        print("Writing temporary document to collection 'system'...")
        doc_ref.set({
            "status": "connected",
            "service": "multimodal-rag-vqa",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
        # 4. Read it back
        print("Reading document back from Firestore...")
        snapshot = doc_ref.get()
        if not snapshot.exists:
            print("FIREBASE CONNECTION: FAIL (Document not found after write)")
            sys.exit(1)
            
        data = snapshot.to_dict()
        print(f"Document read successfully. Data: {data}")
        
        # 5. Verify values
        if data.get("status") == "connected" and data.get("service") == "multimodal-rag-vqa":
            print("\n========================================")
            print("FIREBASE CONNECTION: PASS")
            print("========================================\n")
        else:
            print("FIREBASE CONNECTION: FAIL (Mismatched content)")
            sys.exit(1)
            
    except Exception as e:
        print(f"FIREBASE CONNECTION: FAIL (Exception occurred: {e})")
        sys.exit(1)
    finally:
        # 7. Clean up
        try:
            print("Cleaning up temporary test document...")
            doc_ref.delete()
            print("Cleanup completed.")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

if __name__ == "__main__":
    main()
