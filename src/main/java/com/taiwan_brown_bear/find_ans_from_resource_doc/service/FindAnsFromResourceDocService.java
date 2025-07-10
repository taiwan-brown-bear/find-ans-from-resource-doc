package com.taiwan_brown_bear.find_ans_from_resource_doc.service;

import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import com.taiwan_brown_bear.find_ans_from_resource_doc.configuration.FindAnsFromResourceDocConfiguration;
import com.taiwan_brown_bear.find_ans_from_resource_doc.service.init.FindAnsFromResourceDocInitialization;
import com.taiwan_brown_bear.find_ans_from_resource_doc.service.translator.FindAnsFromResourceDocBertTranslator;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Paths;

@Slf4j
@Service
public class FindAnsFromResourceDocService {

    private FindAnsFromResourceDocConfiguration  conf;
    private FindAnsFromResourceDocInitialization init;
    private FindAnsFromResourceDocBertTranslator translator;

    public FindAnsFromResourceDocService(FindAnsFromResourceDocConfiguration conf) throws IOException {
        this.conf = conf;
        log.info("Initializing / Downloading ...");
        this.init = new FindAnsFromResourceDocInitialization(conf);
        this.translator = new FindAnsFromResourceDocBertTranslator(conf);
    }

    public String findAnswer(String providedResourceDoc, String question)
    {
        try
        {
            // Note: use criteria to get model and,
            //       then, use model to get predictor.
            //       both criteria and predictor need translator.
            //
            //FindAnsFromResourceDocBertTranslator translator = new FindAnsFromResourceDocBertTranslator();

            Criteria<QAInput, String> modelConfigAndSearchCriteria = Criteria.builder()
                    .setTypes(QAInput.class, String.class)
                    .optModelPath(Paths.get(conf.getLOCAL_VOCABULARY_PYTORCH_MODEL_DIRECTORY_URL()))// search this local folder
                    .optTranslator(translator)
                    .optProgress(new ProgressBar()).build();

            log.info("start downloading the model ...");
            ZooModel deserializedPytorchModel = modelConfigAndSearchCriteria.loadModel();
            log.info("finished downloading the model ...");

            try (Predictor<QAInput, String> predictor = deserializedPytorchModel.newPredictor(translator)) {
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
}
