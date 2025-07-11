package com.taiwan_brown_bear.find_ans_from_resource_doc.service;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import com.taiwan_brown_bear.find_ans_from_resource_doc.configuration.FindAnsFromResourceDocConfiguration;
import com.taiwan_brown_bear.find_ans_from_resource_doc.service.init.FindAnsFromResourceDocInitialization;
import com.taiwan_brown_bear.find_ans_from_resource_doc.service.translator.FindAnsFromResourceDocBertTranslator;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

@Slf4j
@Service
public class FindAnsFromResourceDocService {

    private FindAnsFromResourceDocInitialization init;
    private FindAnsFromResourceDocBertTranslator translator;
    private ZooModel                             model;

    public FindAnsFromResourceDocService(FindAnsFromResourceDocConfiguration conf) throws IOException, MalformedModelException, ModelNotFoundException {
        this.init       = new FindAnsFromResourceDocInitialization(conf);
        this.translator = new FindAnsFromResourceDocBertTranslator(conf);
        this.model      = getModel(Paths.get(conf.getLOCAL_VOCABULARY_PYTORCH_MODEL_DIRECTORY_URL()), translator);
    }

    public String findAnswer(String providedResourceDoc, String question)
    {
        try {
            try (Predictor<QAInput, String> predictor = model.newPredictor(translator)) {
                QAInput input = new QAInput(question, providedResourceDoc);
                String answer = predictor.predict(input);
                log.info("QAInput has Question as \"{}\" & Resource Document as \"{}\"", input.getQuestion(), input.getParagraph() );
                log.info("Answer  is \"{}\"", answer);
                return answer;
            }
        } catch (Exception e) {
            log.error("Failed to answer due to {}", e.getMessage(), e);
        }
        return null;
    }

    private ZooModel getModel(Path modelPath, FindAnsFromResourceDocBertTranslator translator)
            throws
            IOException,
            ai.djl.repository.zoo.ModelNotFoundException,
            ai.djl.MalformedModelException {

        // Note: use criteria to get model and,
        //       then, use model to get predictor.
        //       both criteria and predictor need translator.
        //
        Criteria<QAInput, String> modelConfigAndSearchCriteria = Criteria.builder()
                .setTypes(QAInput.class, String.class)
                .optModelPath(modelPath)// search this local folder
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        return modelConfigAndSearchCriteria.loadModel();
    }
}
