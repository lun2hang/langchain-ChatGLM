import os
import logging
import torch
import datetime
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddlePDFLoader
from textsplitter import ChineseTextSplitter
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS


#初始化日志
LOG_FORMAT = "%(levelname) -5s %(asctime)s" " : %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
#logger 自检
logger.info("logger ok")

#Embedding可用集合，name到对应path 配置
embedding_model_dict = {
    "text2vec-large-chinese": "/workspace/chatpdf/models/text2vec-large-chinese",
    "model-name": "model-path",
}
# Embedding model 配置
EMBEDDING_MODEL = "text2vec-large-chinese"
# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


#是否使用预先生成向量索引库，如配置使用预先的建库，文件路径将被忽略；如不配置，使用文件路径建库
#vs_path = ""
vs_path = "/workspace/chatpdf/langchain-ChatGLM/knowledge_base/samples/upload_lun/xajh_FAISS_20230903_153454/vector_store"
#输入文件
filepath = "/workspace/chatpdf/langchain-ChatGLM/knowledge_base/samples/upload_lun/xajh.pdf"
#文本分句长度每行大于100时触发再截断
SENTENCE_SIZE = 100
def load_file(filepath, sentence_size=SENTENCE_SIZE):
    if filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    else:
        logger.info("only support txt csv pdf: %s" % filepath)
    write_check_file(filepath, docs)
    return docs
def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'loaded_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()

#初始化sentence transformer（推理用而非存储推理结果），使用上面配置的预训练语言模型
embedder = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                   model_kwargs={'device': EMBEDDING_DEVICE})

#使用embber 和faiss 建向量库   
if vs_path and os.path.isdir(vs_path) and "index.faiss" in os.listdir(vs_path):
    logger.info("正在使用已建索引文件，生成向量库")
    #从路径加载索引
    vector_store = FAISS.load_local(vs_path, embedder)
else:
    logger.info("正在使用输入文件，生成向量库")
    if not vs_path:
        # 知识库默认存储路径
        KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
        vs_path = os.path.join(KB_ROOT_PATH,
                                f"""{"".join(lazy_pinyin(os.path.splitext(filepath)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",
                                "vector_store")
    #从文件建立索引
    docs = load_file(filepath, sentence_size = SENTENCE_SIZE)   
    vector_store = FAISS.from_documents(docs, embedder)  # docs 为Document列表
    #并保存索引
    vector_store.save_local(vs_path)

#搜索查询
while True:
    query = input("Input your question 请输入问题：")
#    query = "林平之"
    vector_store.chunk_size = 100
    vector_store.score_threshold = 700  
    related_docs_with_score = vector_store.similarity_search_with_score(query, k = 4)
    context = "\n".join([doc.page_content for doc , _ in related_docs_with_score])
    logger.info("query: %s \n result: %s\n" %(query, context))