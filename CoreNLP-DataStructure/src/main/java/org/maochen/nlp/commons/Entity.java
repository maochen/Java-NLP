package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * Created by Maochen on 10/15/15.
 */
public class Entity<T> extends ArrayList<T> {
    public UUID id = UUID.randomUUID();
    public String suggestedName = null;
    public Set<String> attributes = new HashSet<>();
    public Set<BinRelation> binRelations = new HashSet<>();

    public Entity() {

    }

    public Entity(T t) {
        this.add(t);
    }

    @Override
    public String toString() {
        if (suggestedName != null) {
            return suggestedName;
        } else {
            return this.stream().map(Object::toString).reduce((x1, x2) -> x1 + StringUtils.SPACE + x2).orElse(StringUtils.EMPTY);
        }
    }
}
