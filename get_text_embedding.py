import dashscope
import numpy as np
from http import HTTPStatus
import pandas as pd
import os

# 确保 Dashscope API 密钥已正确设置
dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = dashscope_api_key


#读取CCPdoc_embedded.pkl文件
CCPdoc=pd.read_pickle('CCPdoc_embedded.pkl')

# 将 'vector' 列中的所有向量堆叠成一个大的 NumPy 数组
CCPdoc_vector = np.stack(CCPdoc['vector'].to_numpy())





def load_keywords(keyword_filepath):
    keywords = []
    with open(keyword_filepath, 'r', encoding='utf-8') as f:  # 使用合适的编码打开文件
        for line in f:
            keywords.append(line.strip())
    return keywords


def replace_keywords(text, keywords):
    for word in keywords:
        if word in text:
            text = text.replace(word, 'X' * len(word))
    return text

def embed_text(text):
    response = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v2,
        input=text)

    if response.status_code == HTTPStatus.OK:
        # 使用键访问来获取嵌入向量
        embeddings = response.output['embeddings'][0]['embedding']
        return np.array(embeddings)
    else:
        raise Exception(f"Error in text embedding: {response}")

def process_excel_and_save(filepath):
    df = pd.read_excel(filepath)

    # 合并指定列的文本
    df['combined_text'] = df[['headline', 'subheadline', 'title1', 'title2', 'title3', 'content']].apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1)

    # 应用 embed_text 函数
    df['vector'] = df['combined_text'].apply(embed_text)

    #读取keyword.txt文件
    keywords =load_keywords('keyword.txt')
    #如果df['combined_text'] 里面有keyword，就把对应的文字按照字数变成对应数量的X,比如“习近平”就变成“XXX”
    df['combined_text'] = df['combined_text'].apply(lambda x: replace_keywords(x, keywords))

    #将df['combined_text'] 里面的 'nan' 替换成空字符串
    df['combined_text']=df['combined_text'].replace('nan', '', regex=True)

    # 保存到 pickle 文件
    pickle_filepath = filepath.replace('.xlsx', '_embedded.pkl')
    df.to_pickle(pickle_filepath)
    print(f"Processed data saved to {pickle_filepath}")


#对query做处理，加上相关的参考文献
def query_process(query,Thread_value=0.6,max_return=2):
    embed_query=embed_text(query)
    # 计算余弦相似度
    cosine_similarities = np.dot(CCPdoc_vector, embed_query.T) / (
            np.linalg.norm(CCPdoc_vector, axis=1) * np.linalg.norm(embed_query))

    # 找到相似度大于阈值 Thread_value 的索引
    above_threshold_indices = np.where(cosine_similarities > Thread_value)[0]

    # 从这些索引中找到相似度最高的 max_return 个索引
    most_related_index = above_threshold_indices[
        np.argsort(cosine_similarities[above_threshold_indices])[::-1][:max_return]]

    # 找到相似度最高的 max_return 个索引对应的文档
    most_related_docs = CCPdoc['combined_text'].iloc[most_related_index]


    # 构建更新后的查询字符串
    reference_texts = ["参考文献{}: {}".format(i+1, doc) for i, doc in enumerate(most_related_docs)]
    query_updated = ('请根据要求，并借鉴参考借鉴文档的格式和书写形式，除非用户提出特定要求，为国网湖北省电力有限公司写出符合要求的文章。如果要求不明确则直接根据参考资料续写文章。'
                     '注意XXXX部分需要自行补充\n要求为：{}\n{}').format(query, '\n'.join(reference_texts)) if len(reference_texts) > 0 else ('请根据要求，并借鉴参考文档的格式和书写形式，写出符合要求的文章。如果要求不明确则直接根据参考资料续写文章。注意XXXX部分需要自行补充\n要求为：{}\n'.format(query))

    return query_updated


# from Qwen_fortest import model_chat
# import pandas as pd
#
# def test_for_sensitive_words(df):
#     sensitive_rows = []
#     for index, row in df.iterrows():
#         try:
#             # 调用 model_chat 函数
#             next(model_chat(row['combined_text'], None, 'You are a helpful assistant.'))
#         except Exception as e:
#             print(f"Error at row {index}: {e}")
#             sensitive_rows.append(index)
#     return sensitive_rows



if __name__ == '__main__':

    # testvt=embed_text('衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买')

    query_process('我们要抓实抓牢安全生产，以习近平新时代特色社会主义思想为知道，不忘初心，牢记使命',Thread_value=0.6)

    # try:
    #     process_excel_and_save('CCPdoc.xlsx')
    # except Exception as e:
    #     print("An error occurred:", e)

