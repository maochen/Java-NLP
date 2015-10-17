package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Created by Maochen on 10/17/15.
 */
public class TupleRelation<T> {
    public UUID id = UUID.randomUUID();

    private String rel = StringUtils.EMPTY;

    private String relType = StringUtils.EMPTY;

    private List<Entity<T>> entities = new ArrayList<>();

    public String getRel() {
        return rel;
    }

    public void setRel(String rel) {
        this.rel = rel;
    }

    public String getRelType() {
        return relType;
    }

    public void setRelType(String relType) {
        this.relType = relType;
    }

    public List<Entity<T>> getEntities() {
        return entities;
    }

    public void setEntities(List<Entity<T>> entities) {
        this.entities = entities;
    }
}
