<b>Project Overview:<b/>

This repository contains an NLP-based RAG (Retrieval-Augmented Generation) Marketing Content Generator 
that utilizes Amazon Reviews from the ‘All Beauty’ category and corresponding metadata to generate marketing content. 
The system retrieves relevant customer reviews and product details using semantic search and enhances content generation through a generative language model.

Features :

✅ Retrieval-Augmented Generation (RAG) Pipeline
	•	Uses Sentence Transformers to retrieve relevant customer reviews.
	•	Stores embeddings in a vector database 
	•	Filters results based on sentiment, keywords, and product metadata.

✅ Content Generation
	•	Generates product descriptions, social media ads, and email marketing content.
	•	Uses transformer model for high-quality content creation.
	•	Supports tone adjustment (formal, casual, enthusiastic).

✅ Dataset
	•	Utilizes Amazon Reviews (‘All Beauty’ category) and corresponding metadata.
	•	Reviews contain insights into customer experiences, product performance, and usability.
	•	Metadata includes product name, brand, price, rating, and key features.

✅ Customization & Adaptability
	•	Supports multi-platform content formatting (e.g., Blog, Twitter, Email, Ad , Jingles).
	•	Allows parameter tuning for different audience targeting.
