package com.taiwan_brown_bear.find_ans_from_resource_doc.service.init;

import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import com.taiwan_brown_bear.find_ans_from_resource_doc.configuration.FindAnsFromResourceDocConfiguration;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;

@Slf4j
public class FindAnsFromResourceDocInitialization {

    public FindAnsFromResourceDocInitialization(FindAnsFromResourceDocConfiguration conf) throws IOException {

        log.info("(1/2) downloading the vocabulary file to local (Note: In vocab.txt, each row is one word.)");
        final String LOCAL_VOCABULARY_FILE_URL = conf.getLOCAL_VOCABULARY_PYTORCH_MODEL_DIRECTORY_URL() + conf.getLOCAL_VOCABULARY_FILE_NAME();
        DownloadUtils.download(conf.getVOCABULARY_FILE_S3_URL(), LOCAL_VOCABULARY_FILE_URL, new ProgressBar());

        log.info("(2/2) downloading the pytorch model file to local (Note: We are calling python model from JAVA code.)");
        final String LOCAL_PYTORCH_FILE_URL = conf.getLOCAL_VOCABULARY_PYTORCH_MODEL_DIRECTORY_URL() + conf.getLOCAL_PYTORCH_MODEL_FILE_NAME();
        DownloadUtils.download(conf.getPYTORCH_MODEL_FILE_URL(), LOCAL_PYTORCH_FILE_URL, new ProgressBar());

    }
}
