package com.taiwan_brown_bear.find_ans_from_resource_doc.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder(toBuilder = true)
public class FindAnsFromResourceDocResponse {
    private String answer;
}
