<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Project - README</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5; /* Light gray background */
            color: #333;
        }
        .container {
            max-width: 960px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: #fff;
            border-radius: 0.75rem; /* rounded-xl */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            font-weight: 700; /* bold */
            color: #2c5282; /* dark blue */
            margin-bottom: 1rem;
        }
        h1 { font-size: 2.5rem; /* text-4xl */ }
        h2 { font-size: 2rem; /* text-3xl */ border-bottom: 2px solid #edf2f7; padding-bottom: 0.5rem; margin-top: 2rem;}
        h3 { font-size: 1.5rem; /* text-2xl */ margin-top: 1.5rem;}
        p {
            margin-bottom: 1rem;
            line-height: 1.6;
        }
        ul {
            list-style-type: disc;
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }
        ul ul {
            list-style-type: circle;
            margin-left: 1.5rem;
        }
        pre {
            background-color: #e2e8f0; /* gray-200 */
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin-bottom: 1rem;
        }
        code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
            background-color: #edf2f7; /* gray-100 */
            padding: 0.2em 0.4em;
            border-radius: 0.25rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }
        th, td {
            border: 1px solid #e2e8f0; /* gray-200 */
            padding: 0.75rem;
            text-align: left;
        }
        th {
            background-color: #f7fafc; /* gray-50 */
            font-weight: 600;
        }
        td code {
            background-color: transparent;
            padding: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-6 text-indigo-700">Sentiment Analysis Project</h1>

        <p class="text-lg mb-4">
            This repository hosts a Jupyter Notebook dedicated to sentiment analysis. The project aims to classify text into positive, negative, or neutral sentiments by employing both <strong>classical machine learning techniques</strong> with TF-IDF vectorization and <strong>advanced deep learning methods</strong> leveraging BERT embeddings.
        </p>

        <h2 class="text-blue-700">🚀 Project Overview</h2>
        <p>
            This project outlines a <strong>comprehensive sentiment analysis pipeline</strong>. It evaluates <em>traditional machine learning models</em> using TF-IDF features and integrates the <em>Hugging Face Transformers library</em> to utilize pre-trained BERT models for <strong>sophisticated text representation</strong>. The core objective is to benchmark the performance of these diverse approaches in sentiment classification tasks.
        </p>

        <h2 class="text-blue-700">✨ Features</h2>
        <ul>
            <li><strong>Data Handling:</strong> Seamless loading of <code>train.csv</code> and <code>test.csv</code> datasets.</li>
            <li><strong>Robust Null Value Management:</strong> Custom function for intelligent handling of missing values based on their percentage in the dataset.</li>
            <li><strong>Comprehensive Text Preprocessing:</strong> Includes functionalities for:
                <ul>
                    <li>Lowercasing text</li>
                    <li>Removal of custom English stopwords (preserving negations)</li>
                    <li>Stripping URLs</li>
                    <li>Eliminating punctuation</li>
                    <li>Normalizing (removing) extra spaces</li>
                </ul>
            </li>
            <li><strong>Insightful Exploratory Data Analysis (EDA):</strong> Visualizations covering:
                <ul>
                    <li>Distribution of tweet lengths per sentiment category.</li>
                    <li>Sentiment label distribution across tweet times and user age groups.</li>
                    <li>Identification of <strong>most frequent words</strong> in positive, negative, and neutral tweets.</li>
                </ul>
            </li>
            <li><strong>TF-IDF Vectorization:</strong> Transforms raw text data into numerical feature vectors using <code>TfidfVectorizer</code>.</li>
            <li><strong>Classical Machine Learning Models:</strong> Implementation and evaluation of:
                <ul>
                    <li><strong>Logistic Regression</strong></li>
                    <li><strong>Multinomial Naive Bayes</strong></li>
                    <li><strong>Linear SVM</strong></li>
                    <li><strong>Random Forest</strong></li>
                </ul>
            </li>
            <li><strong>Performance Metrics:</strong> Provides detailed <em>accuracy scores</em>, confusion matrices, and classification reports for classical ML models.</li>
            <li><strong>BERT Embeddings Integration:</strong> PyTorch <code>Dataset</code> classes (<code>TextDatasetWithEmbeddings</code>, <code>TokenizedDataset</code>) are included for:
                <ul>
                    <li>Preparing data for pre-trained BERT models.</li>
                    <li>Pre-computing BERT embeddings in batches for optimized GPU memory utilization.</li>
                </ul>
            </li>
        </ul>

        <h2 class="text-blue-700">⚙️ Installation</h2>
        <p>To set up the project locally, follow these steps:</p>
        <ol>
            <li><strong>Clone the repository:</strong>
                <pre><code class="language-bash">git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
</code></pre>
            </li>
            <li><strong>Install dependencies:</strong>
                <pre><code class="language-bash">pip install pandas numpy nltk scikit-learn matplotlib seaborn torch transformers
</code></pre>
            </li>
            <li><strong>Download NLTK stopwords:</strong>
                <pre><code class="language-python">import nltk
nltk.download('stopwords')
</code></pre>
            </li>
        </ol>

        <h2 class="text-blue-700">📊 Dataset</h2>
        <p>The project relies on two CSV files: <code>train.csv</code> and <code>test.csv</code>. These datasets should be placed in the <strong>project's root directory</strong>. The notebook verifies their presence at runtime.</p>
        <p>The <code>train_df</code> is expected to contain columns such as <code>textID</code>, <code>text</code>, <code>selected_text</code>, <code>sentiment</code>, <code>Time of Tweet</code>, <code>Age of User</code>, <code>Country</code>, <code>Population -2020</code>, <code>Land Area (Km²)</code>, and <code>Density (P/Km²)</code>.</p>

        <h2 class="text-blue-700">🧹 Data Preprocessing</h2>
        <p>The <code>preprocess_text</code> function executes a series of <em>text cleaning operations</em>: lowercasing, removal of stopwords (with negation retention), URL stripping, punctuation removal, and extra space normalization.</p>
        <p>Missing values are managed by the <code>handle_null_values</code> function. Rows with less than 5% missing values are removed; otherwise, rows are retained based on crucial columns like "text" and "sentiment".</p>

        <h2 class="text-blue-700">🧠 Models</h2>

        <h3>Classical Machine Learning Models</h3>
        <p><code>TfidfVectorizer</code> is applied to transform processed text into numerical features, which are then used to train and evaluate the following models:</p>
        <ul>
            <li><strong>Logistic Regression</strong></li>
            <li><strong>Multinomial Naive Bayes</strong></li>
            <li><strong>Linear SVM</strong></li>
            <li><strong>Random Forest</strong></li>
        </ul>

        <h3>Deep Learning Data Preparation</h3>
        <p>For deep learning methodologies, <code>BertTokenizer</code> and <code>BertModel</code> are utilized to generate embeddings. The <code>TextDatasetWithEmbeddings</code> class handles the batch pre-computation of BERT embeddings to optimize memory, while <code>TokenizedDataset</code> prepares raw tokenized inputs for model consumption.</p>

        <h2 class="text-blue-700">📈 Results</h2>
        <p>Initial performance of the classical machine learning models on the test set:</p>
        <ul>
            <li><strong>Logistic Regression:</strong> <strong>70.49%</strong></li>
            <li><strong>Naive Bayes:</strong> <strong>63.38%</strong></li>
            <li><strong>Linear SVM:</strong> <strong>69.16%</strong></li>
            <li><strong>Random Forest:</strong> <strong>64.88%</strong></li>
        </ul>
        <p>The notebook also provides *confusion matrices* and comprehensive classification reports for both test and validation datasets. Visualizations of word frequencies and sentiment distributions offer deeper insights into the dataset's characteristics.</p>

        <h2 class="text-blue-700">🚀 Usage</h2>
        <p>To execute the sentiment analysis:</p>
        <ol>
            <li>Ensure <code>train.csv</code> and <code>test.csv</code> are in your project's root.</li>
            <li>Open the <code>Sentiment (2) (2).ipynb</code> notebook in a Jupyter environment (e.g., Google Colab, JupyterLab).</li>
            <li>Run all cells sequentially to initiate data loading, preprocessing, model training, and evaluation.</li>
        </ol>

        <h2 class="text-blue-700">🤝 Contributing</h2>
        <p>Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests for any enhancements or bug fixes.</p>

        <h2 class="text-blue-700">📄 License</h2>
        <p>This project is licensed under the MIT License.</p>

        <hr class="my-8 border-t-2 border-gray-200">

        <h2 class="text-blue-700">Markdown Styling Examples</h2>
        <p>This section demonstrates various text styling options available in Markdown for <code>.md</code> files and comment fields.</p>

        <h3>Headings</h3>
        <h1>A first-level heading</h1>
        <h2>A second-level heading</h2>
        <h3>A third-level heading</h3>

        <h3>Styling Text</h3>
        <div class="overflow-x-auto">
            <table class="min-w-full bg-white rounded-lg shadow-sm">
                <thead>
                    <tr>
                        <th class="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm leading-4 font-medium text-gray-600 uppercase tracking-wider">Style</th>
                        <th class="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm leading-4 font-medium text-gray-600 uppercase tracking-wider">Syntax</th>
                        <th class="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm leading-4 font-medium text-gray-600 uppercase tracking-wider">Example</th>
                        <th class="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm leading-4 font-medium text-gray-600 uppercase tracking-wider">Output</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><strong>Bold</strong></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>** **</code> or <code>__ __</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>**This is bold text**</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><strong>This is bold text</strong></td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><em>Italic</em></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>* *</code> or <code>_ _</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>_This text is italicized_</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><em>This text is italicized</em></td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><s>Strikethrough</s></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>~~ ~~</code> or <code>~ ~</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>~~This was mistaken text~~</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><s>This was mistaken text</s></td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><strong>Bold and nested italic</strong></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>** **</code> and <code>_ _</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>**This text is _extremely_ important**</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><strong>This text is <em>extremely</em> important</strong></td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><strong><em>All bold and italic</em></strong></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>*** ***</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>***All this text is important***</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><strong><em>All this text is important</em></strong></td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm">Subscript</td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>&lt;sub&gt; &lt;/sub&gt;</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>This is a &lt;sub&gt;subscript&lt;/sub&gt; text</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm">This is a <sub>subscript</sub> text</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm">Superscript</td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>&lt;sup&gt; &lt;/sup&gt;</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>This is a &lt;sup&gt;superscript&lt;/sup&gt; text</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm">This is a <sup>superscript</sup> text</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><u>Underline</u></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>&lt;ins&gt; &lt;/ins&gt;</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><code>This is an &lt;ins&gt;underlined&lt;/ins&gt; text</code></td>
                        <td class="py-3 px-4 border-b border-gray-200 text-sm"><u>This is an underlined text</u></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
