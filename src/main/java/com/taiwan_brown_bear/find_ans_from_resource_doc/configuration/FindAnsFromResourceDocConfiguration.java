package com.taiwan_brown_bear.find_ans_from_resource_doc.configuration;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Getter
@Configuration
public class FindAnsFromResourceDocConfiguration {

    @Value("${vocabulary.file.url}")
    private String VOCABULARY_FILE_S3_URL;
    @Value("${pytorch.model.file.url}")
    private String PYTORCH_MODEL_FILE_URL;
    @Value("${local.vocabulary.pytorch.directory.url}")
    private String LOCAL_VOCABULARY_PYTORCH_MODEL_DIRECTORY_URL;
    @Value("${local.vocabulary.file.name}")
    private String LOCAL_VOCABULARY_FILE_NAME;
    @Value("${local.pytorch.file.name}")
    private String LOCAL_PYTORCH_MODEL_FILE_NAME;

}