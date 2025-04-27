<h1>Retrieval-Augmented Product Recommendation System (MSc Project)</h1>

<h2>📚 Project Overview</h2>
<p>This project focuses on building and evaluating a Retrieval-Augmented Generation (RAG) based product recommendation system.
The emphasis is on optimizing <strong>retrieval strategies</strong>, rather than generation, to improve early-ranking performance using real-world product search data.</p>

<p>We explore:</p>
<ul>
<li><strong>Semantic retrieval</strong> (using dense embeddings)</li>
<li><strong>Hybrid retrieval</strong> (combining lexical and semantic filtering)</li>
<li><strong>Similarity function comparisons</strong> (cosine, inner product, L2)</li>
<li><strong>Embedding model comparisons</strong> (MiniLM vs QANet)</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>
<table>
<thead>
<tr><th>Folder/File</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td><code>src/generation/</code></td><td>Code for generation module (e.g., Llama 2 based output creation)</td></tr>
<tr><td><code>src/hybrid_search/</code></td><td>Hybrid retrieval pipeline combining lexical and semantic search</td></tr>
<tr><td><code>src/retrieval/</code></td><td>Semantic retrieval logic, vector search operations, scoring</td></tr>
<tr><td><code>src/utils/</code></td><td>Utility functions for timing, logging, MLflow metric tracking</td></tr>
<tr><td><code>src/data_loader.py</code></td><td>Scripts for loading and preprocessing product metadata and queries</td></tr>
<tr><td><code>tests/</code></td><td>Containes automated tests and evaluations</td></tr>
<tr><td><code>main.py</code></td><td>Main file for running the retrieval and evaluation pipeline</td></tr>
<tr><td><code>mlflow_server.sh</code></td><td>Shell script to start MLflow tracking server locally</td></tr>
<tr><td><code>requirements.txt</code></td><td>Python package dependencies for setting up the project</td></tr>
<tr><td><code>Reports/</code></td><td>Final project report and supporting documentation</td></tr>
<tr><td><code>README.md</code></td><td>Project summary, installation steps, and usage guide</td></tr>
</tbody>
</table>

<hr>

<h2>🛠️ Tools & Libraries</h2>
<ul>
<li><strong>SentenceTransformers</strong> – Semantic embedding generation</li>
<li><strong>ChromaDB</strong> – Dense vector store for efficient similarity search</li>
<li><strong>Whoosh</strong> – Lexical search engine for keyword filtering</li>
<li><strong>MLflow</strong> – Experiment tracking and performance logging</li>
<li><strong>scikit-learn, scipy</strong> – Evaluation metrics, correlation analysis</li>
<li><strong>Matplotlib, Seaborn</strong> – Data visualization</li>
</ul>

<hr>

<h2>📊 Experiments & Metrics</h2>
<p>Key experiments conducted:</p>
<ul>
<li><strong>Precision@K</strong>, <strong>Recall@K</strong>, <strong>MRR</strong> evaluation for each retrieval strategy</li>
<li><strong>Impact of similarity function</strong> on retrieval performance</li>
<li><strong>Embedding model correlation</strong> analysis (Spearman rank)</li>
</ul>

<p>MLflow dashboard link for full experiment results:<br>
👉 <a href="#">http://localhost:5000/#/experiments/0?viewStateShareKey=60b5838b3c07b10d688fd52b4dd6c37593b139dcfb12d21877e12fcb552682f6</a></p>

<hr>

<h2>📝 Dataset Used</h2>
<ul>
<li><strong>Amazon ESCI Dataset (ESCI-small)</strong>: Real-world Amazon search queries with labeled ground truth products.</li>
<li>Preprocessing included filtering English queries and metadata cleaning.</li>
</ul>

<p>Link to dataset source:<br>
🔗 <a href="https://github.com/hyp1231/AmazonReviews2023">Amazon Reviews 2023 - ESCI Product Search Dataset</a></p>



<h2>📈 Key Results</h2>
<ul>
<li>Hybrid retrieval slightly outperformed pure semantic retrieval at lower K values.</li>
<li>Minimal performance gap between MiniLM and QANet embeddings.</li>
<li>Spearman correlation between model similarity scores: <strong>0.83</strong> (high agreement).</li>
</ul>

<hr>

<h2>✅ Future Work</h2>
<ul>
<li>Fusion Reranking(In progress)</li>
<li>Experiment with larger, more diverse query datasets.</li>
<li>Fine-tune embedding models for domain-specific retrieval.</li>
<li>Explore lightweight generation models for recommendation summaries.</li>
</ul>

<hr>

<h2>📜 References</h2>
<ul>
<li>Lewis et al., 2020. <em>Retrieval-Augmented Generation (RAG)</em>.</li>
<li>Reimers and Gurevych, 2019. <em>Sentence-BERT for dense retrieval</em>.</li>
<li>Hou et al., 2023. <em>Amazon Reviews 2023 and ESCI Dataset</em>.</li>
</ul>
