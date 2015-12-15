package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Created by Maochen on 10/17/15.
 */
public class TupleRelation {
    public UUID id = UUID.randomUUID();

    private String rel = StringUtils.EMPTY;

    public String relType = StringUtils.EMPTY; // Can be VerbNet roleset or freebase

    private List<Entity> entities = new ArrayList<>();

    public Map<String, String> feats = new HashMap<>();

    public String getRel() {
        return rel;
    }

    public void setRel(String rel) {
        this.rel = rel;
    }

    public List<Entity> getEntities() {
        return entities;
    }

    public void setEntities(List<Entity> entities) {
        this.entities = entities;
    }
}
