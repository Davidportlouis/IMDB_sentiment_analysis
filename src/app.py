#! /bin/python3

import os
import torch
import flask
import numpy
import transformers
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restplus import Api, Resource
from flask import Flask, request,  Markup

app = Flask(__name__)
api = Api(app, version='1.0-beta', title="Xinxi")

# configs
BERT_PATH = "./models/bert_base_uncased/"
DISTIL_PATH = "./models/distilbert_base_uncased/"
ROBERTA_PATH = "./models/roberta_base/"
MAX_LEN = 512
MODEL = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_emoji(sentiment):
    if sentiment == "positive":
        return "\U0001F60A"
    elif sentiment == "negative":
        return "\U0001F613"
    elif sentiment == "neutral":
        return "\U0001F610"

        
def bert_predict(model, input_):
    tokenizer = transformers.BertTokenizer.from_pretrained(
        BERT_PATH, do_lower_case=True)
    review = str(input_)
    review = " ".join(review.split())
    inputs = tokenizer.encode_plus(review, None, add_special_tokens=True,
                                   max_length=MAX_LEN, padding="max_length", return_token_type_ids=True,
                                   return_tensors="pt", truncation=True)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    outputs = model(input_ids=ids, attention_mask=mask, input=token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


def roberta_predict(model, input_):
    tokenizer = transformers.RobertaTokenizerFast.from_pretrained(ROBERTA_PATH)
    review = str(input_)
    review = " ".join(review.split())
    inputs = tokenizer.encode_plus(review, None, add_special_tokens=True, max_length=MAX_LEN,
                                   padding="max_length", return_token_type_ids=True,
                                   return_tensors="pt", truncation=True)

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    outputs = model(input_ids=ids, attention_mask=mask)
    print(torch.softmax(outputs, dim=1))
    _, preds = torch.max(outputs, dim=1)
    conf, _ = torch.max(torch.softmax(outputs, dim=1), dim=1)
    return "neutral" if preds.item() == 0 else ("negative" if preds.item() == 1 else "positive"), conf.item()


def distilbert_predict(model, input_):
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
        DISTIL_PATH)
    review = str(input_)
    review = " ".join(review.split())
    inputs = tokenizer.encode_plus(review, None, add_special_tokens=True, max_length=MAX_LEN,
                                   padding="max_length", return_token_type_ids=True,
                                   return_tensors="pt", truncation=True)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    outputs = model(input_ids=ids, argument_2=mask)
    print(torch.softmax(outputs, dim=1))
    _, preds = torch.max(outputs, dim=1)
    conf, _ = torch.max(torch.softmax(outputs, dim=1), dim=1)
    return "positive" if preds.item() == 0 else "negative", conf.item()


@api.route("/v1/sentiment/bert/<string:sentence>")
@api.doc(params={"sentence": "Sentence to be analysed for"})
class Bert(Resource):
    def get(self, sentence):
        MODEL = torch.jit.load(BERT_PATH + "bert_traced_qt.pt")
        MODEL.eval()
        pos = bert_predict(MODEL, sentence)
        neg = 1.0 - pos
        sentiment = "positive" if pos > neg else "negative"
        emoji = get_emoji(sentiment)
        res = {}
        res["response"] = {
            "sentence": sentence,
            "positive": str(pos),
            "negative": str(neg),
            "sentiment": sentiment,
            "emoji": emoji
        }
        return flask.jsonify(res)


@api.route("/v1/sentiment/distilbert/<string:sentence>")
@api.doc(params={"sentence": "Sentence to be analysed for"})
class DistilBert(Resource):
    def get(self, sentence):
        MODEL = torch.jit.load(DISTIL_PATH + "distilbert_traced_qt.pt")
        MODEL.eval()
        sentiment, conf = distilbert_predict(MODEL, sentence)
        if sentiment == "positive":
            neg_conf = 1.0 - conf
        elif sentiment == "negative":
            pos_conf = 1.0 - conf
        emoji = get_emoji(sentiment)
        res = {}
        res["response"] = {
            "sentence": sentence,
            "sentiment": sentiment,
            "positive": str(conf) if sentiment == "positive" else str(pos_conf),
            "negative": str(conf) if sentiment == "negative" else str(neg_conf),
            "emoji": emoji
        }
        return flask.jsonify(res)


@api.route("/v1/sentiment/roberta/<string:sentence>")
@api.doc(params={"sentence": "Sentence to be analysed for"})
class Roberta(Resource):
    def get(self, sentence):
        MODEL = torch.jit.load(ROBERTA_PATH + "roberta_traced_qt.pt")
        MODEL.eval()
        sentiment, conf = roberta_predict(MODEL, sentence)
        emoji = get_emoji(sentiment)
        res = {}
        res["response"] = {
            "sentence": sentence,
            "sentiment": sentiment,
            "confidence": conf,
            "emoji": emoji
        }
        return flask.jsonify(res)


if __name__ == "__main__":
    app.run()
