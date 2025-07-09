package com.taiwan_brown_bear.find_ans_from_resource_doc;

import org.springframework.boot.SpringApplication;// original
import org.springframework.boot.autoconfigure.SpringBootApplication;// original

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

import ai.djl.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.inference.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;
import ai.djl.repository.zoo.*;
import ai.djl.modality.nlp.*;
import ai.djl.modality.nlp.qa.*;
import ai.djl.modality.nlp.bert.*;

@SpringBootApplication
public class FindAnsFromResourceDocApplication {

	public static void main(String[] args) {
		SpringApplication.run(FindAnsFromResourceDocApplication.class, args);
		qanda();// new
	}

	public static void qanda(){
		System.out.println("hello world 123");

		var question = "When did BBC Japan start broadcasting?";
		var resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
				"Which operated between December 2004 and April 2006.\n" +
				"It ceased operations after its Japanese distributor folded.";

		QAInput input = new QAInput(question, resourceDocument);

		var tokenizer = new BertTokenizer();
		List<String> tokenQ = tokenizer.tokenize(question.toLowerCase());
		List<String> tokenA = tokenizer.tokenize(resourceDocument.toLowerCase());

		System.out.println("Question Token: " + tokenQ);
		System.out.println("Answer Token: " + tokenA);

		BertToken token = tokenizer.encode(question.toLowerCase(), resourceDocument.toLowerCase());
		System.out.println("Encoded tokens: " + token.getTokens());
		System.out.println("Encoded token type: " + token.getTokenTypes());
		System.out.println("Valid length: " + token.getValidLength());

		try {
			DownloadUtils.download(
					"https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/bert-base-uncased-vocab.txt.gz",
					"build/pytorch/bertqa/vocab.txt", new ProgressBar());

			var path = Paths.get("build/pytorch/bertqa/vocab.txt");
			var vocabulary = DefaultVocabulary.builder()
					.optMinFrequency(1)
					.addFromTextFile(path)
					.optUnknownToken("[UNK]")
					.build();

			long index = vocabulary.getIndex("car");
			String token123 = vocabulary.getToken(2482);
			System.out.println("The index of the car is " + index);
			System.out.println("The token of the index 2482 is " + token123);



			DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/trace_bertqa.pt.gz", "build/pytorch/bertqa/bertqa.pt", new ProgressBar());
			BertTranslator translator = new BertTranslator();

			Criteria<QAInput, String> criteria = Criteria.builder()
					.setTypes(QAInput.class, String.class)
					.optModelPath(Paths.get("build/pytorch/bertqa/")) // search in local folder
					.optTranslator(translator)
					.optProgress(new ProgressBar()).build();

			ZooModel model = criteria.loadModel();

			String predictResult = null;
			QAInput input123 = new QAInput(question, resourceDocument);

// Create a Predictor and use it to predict the output
			try (Predictor<QAInput, String> predictor = model.newPredictor(translator)) {
				predictResult = predictor.predict(input123);
			}

			System.out.println(question);
			System.out.println(predictResult);

		} catch (Exception ioe) {
			System.out.println("ERROR ...");
		}

	}
}
