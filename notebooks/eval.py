import openai
import re
import numpy as np
from tqdm import tqdm

from llama_index.evaluation import CorrectnessEvaluator
from llama_index.llms import OpenAI
from llama_index import ServiceContext
import evaluate

def evaluate_retrieval(
    llama_index_retriever, 
    queries, 
    golden_sources
):
    results = []

    for query, expected_source in tqdm(list(zip(queries, golden_sources))):
        retrieved_nodes = llama_index_retriever.retrieve(query)
        retrieved_sources = [node.metadata['drug_link'] for node in retrieved_nodes]
        
        # If our label does not include a section, then any sections on the page should be considered a hit.
        if "#" not in expected_source:
            retrieved_sources = [source.split("#")[0] for source in retrieved_sources]
        
        if expected_source in retrieved_sources:
            is_hit = True
            score = retrieved_nodes[retrieved_sources.index(expected_source)].score
        else:
            is_hit = False
            score = 0.0
        
        result = {
            "is_hit": is_hit,
            "score": score,
            "retrieved": retrieved_sources,
            "expected": expected_source,
            "query": query,
        }
        results.append(result)
    return results


def get_hit_rate(results):
    return np.mean([r["is_hit"] for r in results])

def get_mean_score(results):
    return np.mean([r.score for r in results])


def calculate_metric_score(metric_name,references, predictions):
    metric = evaluate.load(metric_name)
    if metric_name != "bertscore":
        results = metric.compute(predictions=predictions, references=references)
    else:
        results = metric.compute(predictions=predictions, references=references, lang='en')
                
    return results

from statistics import mean
import pandas as pd
from statistics import mean

def generate_metrics_summary(references, prediction_dict):

    results = {}
    rouge_results = {}
    print("Calculating ROUGE Score...")
    for prediction_label, predictions in prediction_dict.items():
        metric_result = calculate_metric_score('rouge', references, predictions)
        rouge_results[prediction_label] = metric_result
    
    # Convert ROUGE scores to DataFrame
    rouge_df = pd.DataFrame(rouge_results).T.reset_index().rename(columns={"index": "System"})
    
    # Add ROUGE DataFrame to results
    results["rouge"] = rouge_df

    bleu_results = {}
    print("Calculating BLEU Score...")
    for prediction_label, predictions in prediction_dict.items():
        metric_result = calculate_metric_score('bleu', references, predictions)
        metric_result.pop("precisions")
        metric_result.pop("brevity_penalty")
        metric_result.pop("length_ratio")
        metric_result.pop("translation_length")
        metric_result.pop("reference_length")
        bleu_results[prediction_label] = metric_result
    
    # Convert BLEU scores to DataFrame
    bleu_df = pd.DataFrame(bleu_results).T.reset_index().rename(columns={"index": "System"})
    
    # Add BLEU DataFrame to results
    results["bleu"] = bleu_df

    bert_results = {}
    print("Calculating BERT Score...")
    for prediction_label, predictions in prediction_dict.items():
        metric_result = calculate_metric_score('bertscore', references, predictions)
        metric_result["average_bertscore_precision"] = mean(metric_result["precision"])
        metric_result["average_bertscore_recall"] = mean(metric_result["recall"])
        metric_result["average_bertscore_f1"] = mean(metric_result["f1"])
        metric_result.pop("precision")
        metric_result.pop("recall")
        metric_result.pop("f1")
        metric_result.pop("hashcode")
        bert_results[prediction_label] = metric_result
    
    # Convert BERTScore scores to DataFrame
    bert_df = pd.DataFrame(bert_results).T.reset_index().rename(columns={"index": "System"})
    
    # Add BERTScore DataFrame to results
    results["bert"] = bert_df

    meteor_results = {}
    print("Calculating METEOR Score...")
    for prediction_label, predictions in prediction_dict.items():
        metric_result = calculate_metric_score('meteor', references, predictions)
        meteor_results[prediction_label] = metric_result
    
    # Convert METEOR scores to DataFrame
    meteor_df = pd.DataFrame(meteor_results).T.reset_index().rename(columns={"index": "System"})
    
    # Add METEOR DataFrame to results
    results["meteor"] = meteor_df

    return results


def generate_human_eval_summary(references_dict, predictions, system_name):

    results = {}
    rouge_results = {}
    print("Calculating ROUGE Score...")
    for reference_label, references in references_dict.items():
        metric_result = calculate_metric_score('rouge', references, predictions)
        rouge_results[reference_label] = metric_result
    
    # Convert ROUGE scores to DataFrame
    rouge_df = pd.DataFrame(rouge_results).T.reset_index().rename(columns={"index": system_name})
    
    # Add ROUGE DataFrame to results
    results["rouge"] = rouge_df

    bleu_results = {}
    print("Calculating BLEU Score...")
    for reference_label, reference in references_dict.items():
        metric_result = calculate_metric_score('bleu', references, predictions)
        metric_result.pop("precisions")
        metric_result.pop("brevity_penalty")
        metric_result.pop("length_ratio")
        metric_result.pop("translation_length")
        metric_result.pop("reference_length")
        bleu_results[reference_label] = metric_result
    
    # Convert BLEU scores to DataFrame
    bleu_df = pd.DataFrame(bleu_results).T.reset_index().rename(columns={"index": system_name})
    
    # Add BLEU DataFrame to results
    results["bleu"] = bleu_df

    bert_results = {}
    print("Calculating BERT Score...")
    for reference_label, reference in references_dict.items():
        metric_result = calculate_metric_score('bertscore', references, predictions)
        metric_result["average_bertscore_precision"] = mean(metric_result["precision"])
        metric_result["average_bertscore_recall"] = mean(metric_result["recall"])
        metric_result["average_bertscore_f1"] = mean(metric_result["f1"])
        metric_result.pop("precision")
        metric_result.pop("recall")
        metric_result.pop("f1")
        metric_result.pop("hashcode")
        bert_results[reference_label] = metric_result
    
    # Convert BERTScore scores to DataFrame
    bert_df = pd.DataFrame(bert_results).T.reset_index().rename(columns={"index": system_name})
    
    # Add BERTScore DataFrame to results
    results["bert"] = bert_df

    meteor_results = {}
    print("Calculating METEOR Score...")
    for reference_label, reference in references_dict.items():
        metric_result = calculate_metric_score('meteor', references, predictions)
        meteor_results[reference_label] = metric_result
    
    # Convert METEOR scores to DataFrame
    meteor_df = pd.DataFrame(meteor_results).T.reset_index().rename(columns={"index": system_name})
    
    # Add METEOR DataFrame to results
    results["meteor"] = meteor_df

    return results
