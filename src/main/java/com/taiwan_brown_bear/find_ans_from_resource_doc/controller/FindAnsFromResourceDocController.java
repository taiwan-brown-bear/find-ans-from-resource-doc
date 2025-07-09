package com.taiwan_brown_bear.find_ans_from_resource_doc.controller;

import com.taiwan_brown_bear.find_ans_from_resource_doc.dto.FindAnsFromResourceDocRequest;
import com.taiwan_brown_bear.find_ans_from_resource_doc.dto.FindAnsFromResourceDocResponse;
import com.taiwan_brown_bear.find_ans_from_resource_doc.service.FindAnsFromResourceDocService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/ans-from-resource-doc")
public class FindAnsFromResourceDocController {

    @Autowired
    private FindAnsFromResourceDocService findAnsFromResourceDocService;

    @GetMapping
    public ResponseEntity<FindAnsFromResourceDocResponse> getAnswer(@RequestBody FindAnsFromResourceDocRequest request){
        String providedDoc = request.getResourceDoc();
        String question    = request.getQuestion();
        log.info("based on the provided doc, \"{}\"", providedDoc);
        log.info("trying to find the answer for question, \"{}\"", question);
        String answer = findAnsFromResourceDocService.findAnswer(providedDoc, question);
        return ResponseEntity.ok(FindAnsFromResourceDocResponse.builder()
                .answer(answer)
                .build());
    }
}
