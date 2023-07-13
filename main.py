# -*- coding: utf-8 -*-

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from bertopic import BERTopic
import hdbscan
import joblib
import json
from nltk.corpus import stopwords
import os
import pandas as pd
import time
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def extract_topic_sizes_by_date(df, opt_params):
    topic_sizes = (
        df.groupby([f"{opt_params['DATE_COLUMN']}", 'TOPIC'])[f"{opt_params['RAW_COLUMN']}"].count().reset_index()
        .rename({f"{opt_params['DATE_COLUMN']}": "Date", "TOPIC": "Topic", f"{opt_params['RAW_COLUMN']}": "Size"},
                axis='columns')
        .sort_values(["Date", "Topic"], ascending=[True, True]))
    return topic_sizes


def tm_bertopic(config_path, model_path=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    opt_params = {"RAW_PATH": config["DATA"]["CSV_PATH"],
                  "DATE_COLUMN": config["DATA"]["DATE_COLUMN"],
                  "RAW_COLUMN": config["DATA"]["RAW_COLUMN"],
                  "STOPWORDS_PATH": config["DATA"]["STOPWORDS_PATH"],
                  "EMBEDDING_TOKEN_NAME": config["EMBEDDING"]["TOKEN_NAME"],
                  "BERTOPIC_LANG": config["BERTOPIC"]["LANGUAGE"],
                  "UMAP_N_NEIGHBORS": config["BERTOPIC"]["UMAP_N_NEIGHBORS"],
                  "UMAP_N_COMPONENTS": config["BERTOPIC"]["UMAP_N_COMPONENTS"],
                  "UMAP_METRIC": config["BERTOPIC"]["UMAP_METRIC"],
                  "UMAP_MIN_DIST": config["BERTOPIC"]["UMAP_MIN_DIST"],
                  "HDBSCAN_MIN_CLUSTER_SIZE": config["BERTOPIC"]["HDBSCAN_MIN_CLUSTER_SIZE"],
                  "HDBSCAN_MIN_SAMPLES": config["BERTOPIC"]["HDBSCAN_MIN_SAMPLES"],
                  "HDBSCAN_METRIC": config["BERTOPIC"]["HDBSCAN_METRIC"],
                  "HDBSCAN_ALPHA": config["BERTOPIC"]["HDBSCAN_ALPHA"],
                  "HDBSCAN_ALGORITHM": config["BERTOPIC"]["HDBSCAN_ALGORITHM"],
                  "HDBSCAN_LEAF_SIZE": config["BERTOPIC"]["HDBSCAN_LEAF_SIZE"],
                  "TFIDF_NGRAM_MIN": config["BERTOPIC"]["TFIDF_NGRAM_MIN"],
                  "TFIDF_NGRAM_MAX": config["BERTOPIC"]["TFIDF_NGRAM_MAX"],
                  "TOPIC_N": config["BERTOPIC"]["TOPIC_N"],
                  "TOPIC_TOP_N_WORDS": config["BERTOPIC"]["TOP_N_WORDS"],
                  "TOPIC_TOP_N_REP_DOCS": config["BERTOPIC"]["TOP_N_REP_DOCS"],
                  "REP_DOCS_TARGET_COLUMN": config["BERTOPIC"]["REP_DOCS_TARGET_COLUMN"],
                  "GRAPH_WIDTH": config["GRAPH"]["WIDTH"],
                  "GRAPH_HEIGHT": config["GRAPH"]["HEIGHT"],
                  "MODEL_PATH": config["RESULT"]["MODEL_PATH"],
                  "TOPIC_INFO_PATH": config["RESULT"]["TOPIC_INFO_PATH"],
                  "TOPIC_DOC_PATH": config["RESULT"]["TOPIC_DOC_PATH"],
                  "TOPIC_REP_DOCS_PATH": config["RESULT"]["TOPIC_REP_DOCS_PATH"],
                  "TOPIC_SIZE_OVER_TIME_PATH": config["RESULT"]["TOPIC_SIZE_OVER_TIME_PATH"],
                  "TOPIC_WORD_SCORES_PATH": config["RESULT"]["TOPIC_WORD_SCORES_PATH"],
                  "HIERARCHICAL_CLUSTERING_PATH": config["RESULT"]["HIERARCHICAL_CLUSTERING_PATH"],
                  "INTERTOPIC_DISTANCE_MAP_PATH": config["RESULT"]["INTERTOPIC_DISTANCE_MAP_PATH"],
                  "DOCUMENTS_AND_TOPICS_PATH": config["RESULT"]["DOCUMENTS_AND_TOPICS_PATH"],
                  "SIMILARITY_MATRIX_PATH": config["RESULT"]["SIMILARITY_MATRIX_PATH"],
                  "TOPICS_OVER_TIME_PATH": config["RESULT"]["TOPICS_OVER_TIME_PATH"]}

    assert opt_params["BERTOPIC_LANG"] in ["english", "korean"]

    os.makedirs("/".join(opt_params["TOPIC_INFO_PATH"].split("/")[:-1]), exist_ok=True)
    os.makedirs("/".join(opt_params["TOPIC_DOC_PATH"].split("/")[:-1]), exist_ok=True)
    os.makedirs("/".join(opt_params["TOPIC_REP_DOCS_PATH"].split("/")[:-1]), exist_ok=True)
    try:
        os.makedirs("/".join(opt_params["TOPIC_SIZE_OVER_TIME_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs("/".join(opt_params["TOPIC_WORD_SCORES_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs("/".join(opt_params["HIERARCHICAL_CLUSTERING_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs("/".join(opt_params["INTERTOPIC_DISTANCE_MAP_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs("/".join(opt_params["DOCUMENTS_AND_TOPICS_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs("/".join(opt_params["SIMILARITY_MATRIX_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs("/".join(opt_params["TOPICS_OVER_TIME_PATH"].split("/")[:-1]), exist_ok=True)
    except FileNotFoundError:
        pass

    raw_df = pd.read_csv(opt_params["RAW_PATH"]).dropna(
        subset=[opt_params["DATE_COLUMN"], opt_params["RAW_COLUMN"]])
    date_lst = raw_df[opt_params["DATE_COLUMN"]].tolist()
    raw_lst = raw_df[opt_params["RAW_COLUMN"]].tolist()

    stopwords_lst = pd.read_csv(opt_params["STOPWORDS_PATH"])["STOPWORDS"].tolist()

    ngram_range = (opt_params["TFIDF_NGRAM_MIN"], opt_params["TFIDF_NGRAM_MAX"])

    if opt_params["BERTOPIC_LANG"] == "english":
        stop_words = list(stopwords.words(opt_params["BERTOPIC_LANG"])) + stopwords_lst
    else:
        stop_words = stopwords_lst

    embedding_model = SentenceTransformer(opt_params["EMBEDDING_TOKEN_NAME"])

    # we add this to remove stopwords that can pollute topics
    vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)

    start_time = time.time()
    print(f"INFO: Embedding using token: {opt_params['EMBEDDING_TOKEN_NAME']}")

    embeddings = embedding_model.encode(raw_lst, show_progress_bar=True)

    end_time = time.time()
    print(f"INFO: Completed! Elapsed Time: {str(round(end_time - start_time, 2))}s")

    if model_path is None:
        umap_model = umap.UMAP(n_neighbors=opt_params["UMAP_N_NEIGHBORS"],
                               n_components=opt_params["UMAP_N_COMPONENTS"],
                               metric=opt_params["UMAP_METRIC"],
                               min_dist=opt_params["UMAP_MIN_DIST"])

        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=opt_params["HDBSCAN_MIN_CLUSTER_SIZE"],
                                        min_samples=opt_params["HDBSCAN_MIN_SAMPLES"],
                                        metric=opt_params["HDBSCAN_METRIC"],
                                        alpha=opt_params["HDBSCAN_ALPHA"],
                                        algorithm=opt_params["HDBSCAN_ALGORITHM"],
                                        leaf_size=opt_params["HDBSCAN_LEAF_SIZE"],
                                        prediction_data=True, gen_min_span_tree=True)

        if opt_params["BERTOPIC_LANG"] == "english":
            model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics=opt_params["TOPIC_N"],
                top_n_words=opt_params["TOPIC_TOP_N_WORDS"],
                language=opt_params["BERTOPIC_LANG"],
                calculate_probabilities=True,
                verbose=True
            )
        else:
            model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics=opt_params["TOPIC_N"],
                top_n_words=opt_params["TOPIC_TOP_N_WORDS"],
                language="multilingual",
                calculate_probabilities=True,
                verbose=True
            )

        # Creating topics
        start_time = time.time()
        print(f"INFO: Creating Topics")

        topics, probs = model.fit_transform(raw_lst, embeddings=embeddings)

        end_time = time.time()
        print(f"INFO: Completed! Elapsed Time: {str(round(end_time - start_time, 2))}s")

        # Save model
        start_time = time.time()
        print(f"INFO: Saving model to {opt_params['MODEL_PATH']}")

        model.save(opt_params["MODEL_PATH"], serialization="safetensors", save_ctfidf=True)

        end_time = time.time()
        print(f"INFO: Completed! Elapsed Time: {str(round(end_time - start_time, 2))}s")
    else:
        # Load model
        start_time = time.time()
        print(f"INFO: Loading model from {model_path}")

        model = BERTopic.load(model_path, embedding_model=opt_params["EMBEDDING_TOKEN_NAME"])

        end_time = time.time()
        print(f"INFO: Completed! Elapsed Time: {str(round(end_time - start_time, 2))}s")

        # Creating topics
        start_time = time.time()
        print(f"INFO: Creating Topics")

        topics, probs = model.transform(raw_lst, embeddings=embeddings)

        end_time = time.time()
        print(f"INFO: Completed! Elapsed Time: {str(round(end_time - start_time, 2))}s")

    start_time = time.time()
    print(f"INFO: Saving Results")

    # CSV: Topic info
    model_info = model.get_topic_info()
    model_info.to_csv(opt_params["TOPIC_INFO_PATH"], sep=",", encoding="utf-8-sig", index=False)

    # CSV: Topic probabilities by document
    probs_max = [max(x) for x in probs]
    topic_df = pd.DataFrame(zip(topics, probs_max, date_lst, raw_lst),
                            columns=["TOPIC", "PROB", f"{opt_params['DATE_COLUMN']}",
                                     f"{opt_params['RAW_COLUMN']}"])
    topic_df.to_csv(opt_params["TOPIC_DOC_PATH"], sep=",", encoding="utf-8-sig", index=False)

    # CSV: Topic representative docs
    # 내장 함수에 직접 접근, library version에 따라 에러 발생 가능성 고려
    topic_rep_docs_dict, _, _, _ = model._extract_representative_docs(c_tf_idf=model.c_tf_idf_,
                                                                      documents=pd.DataFrame(
                                                                          {"Document": raw_lst,
                                                                           "ID": range(len(raw_lst)),
                                                                           "Topic": model.topics_}),
                                                                      topics=model.topic_representations_,
                                                                      nr_repr_docs=opt_params["TOPIC_TOP_N_REP_DOCS"])
    topic_rep_docs_lst = []
    for k, v in topic_rep_docs_dict.items():
        for e in v:
            topic_rep_docs_lst.append([k, e])
    topic_rep_docs_df = pd.DataFrame(topic_rep_docs_lst, columns=["Topic", "Representative_Docs"])

    if opt_params["REP_DOCS_TARGET_COLUMN"] == "":
        topic_rep_docs_df.to_csv(opt_params["TOPIC_REP_DOCS_PATH"], sep=",", encoding="utf-8-sig", index=False)
    else:
        topic_rep_docs_lst = []
        for row_idx in range(topic_rep_docs_df.shape[0]):
            target_col_data = \
                raw_df[raw_df[opt_params["RAW_COLUMN"]].str.contains(topic_rep_docs_df.iloc[row_idx, 1], case=True)][
                    opt_params["REP_DOCS_TARGET_COLUMN"]].tolist()
            if len(target_col_data) == 0:
                continue
            else:
                for e in target_col_data:
                    topic_rep_docs_lst.append([topic_rep_docs_df.iloc[row_idx, 0], e])

        topic_rep_docs_df = pd.DataFrame(topic_rep_docs_lst, columns=["Topic", "Representative_Docs"])
        topic_rep_docs_df.to_csv(opt_params["TOPIC_REP_DOCS_PATH"], sep=",", encoding="utf-8-sig", index=False)

    # CSV: Topic size over time
    if opt_params["TOPIC_SIZE_OVER_TIME_PATH"] != "":
        topic_sizes_over_time = extract_topic_sizes_by_date(topic_df, opt_params)
        topic_sizes_over_time.to_csv(opt_params["TOPIC_SIZE_OVER_TIME_PATH"], sep=",", encoding="utf-8-sig",
                                     index=False)

    # Graph: Topic word scores
    if opt_params["TOPIC_WORD_SCORES_PATH"] != "":
        topic_words_fig = model.visualize_barchart(top_n_topics=len(set(topics)),
                                                   n_words=opt_params["TOPIC_TOP_N_WORDS"])
        topic_words_fig.write_html(opt_params["TOPIC_WORD_SCORES_PATH"])

    # Graph: Hierarchical clustering
    if opt_params["HIERARCHICAL_CLUSTERING_PATH"] != "":
        hier_fig = model.visualize_hierarchy(width=opt_params["GRAPH_WIDTH"], height=opt_params["GRAPH_HEIGHT"])
        hier_fig.write_html(opt_params["HIERARCHICAL_CLUSTERING_PATH"])

    # Graph: Intertopic distance map
    if opt_params["INTERTOPIC_DISTANCE_MAP_PATH"] != "":
        try:
            topic_fig = model.visualize_topics(width=opt_params["GRAPH_WIDTH"], height=opt_params["GRAPH_HEIGHT"])
            topic_fig.write_html(opt_params["INTERTOPIC_DISTANCE_MAP_PATH"])
        except ValueError:
            print("ERROR: Failed drawing Intertopic Distance Map")
        except TypeError:
            print("ERROR: Failed drawing Intertopic Distance Map")

    # Graph: Documents and topics
    if opt_params["DOCUMENTS_AND_TOPICS_PATH"] != "":
        doc_fig = model.visualize_documents(raw_lst, embeddings=embeddings, width=opt_params["GRAPH_WIDTH"],
                                            height=opt_params["GRAPH_HEIGHT"])
        doc_fig.write_html(opt_params["DOCUMENTS_AND_TOPICS_PATH"])

    # Graph: Similarity matrix
    if opt_params["SIMILARITY_MATRIX_PATH"] != "":
        sim_matrix_fig = model.visualize_heatmap(width=opt_params["GRAPH_WIDTH"], height=opt_params["GRAPH_HEIGHT"])
        sim_matrix_fig.write_html(opt_params["SIMILARITY_MATRIX_PATH"])

    # Graph: Topic size over time
    if opt_params["TOPICS_OVER_TIME_PATH"] != "":
        topics_over_time = model.topics_over_time(raw_lst, date_lst)
        topics_over_time_fig = model.visualize_topics_over_time(topics_over_time, width=opt_params["GRAPH_WIDTH"],
                                                                height=opt_params["GRAPH_HEIGHT"])
        topics_over_time_fig.write_html(opt_params["TOPICS_OVER_TIME_PATH"])

    end_time = time.time()
    print(f"INFO: Completed! Elapsed Time: {str(round(end_time - start_time, 2))}s")


def main():
    config_path = "./config/sample_en.json"
    tm_bertopic(config_path)


if __name__ == "__main__":
    main()
