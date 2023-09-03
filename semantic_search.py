import os
import logging
import torch
from langchain.document_loaders import TextLoader, CSVLoader
from loader import UnstructuredPaddlePDFLoader
from textsplitter import ChineseTextSplitter

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

#使用未分割文件建库
#输入文件
filepath = "/workspace/chatpdf/langchain-ChatGLM/knowledge_base/samples/upload_lun/xajh.pdf"
#文本分句长度
SENTENCE_SIZE = 100
#
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

load_file(filepath, sentence_size = SENTENCE_SIZE)

