package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Created by Maochen on 10/15/15.
 */
public class Entity<T> extends ArrayList<T> {
    public UUID id = UUID.randomUUID();
    public String suggestedName = null;
    public Set<String> type = new HashSet<>();

    public Set<BinRelation> relations = new HashSet<>();
    public Set<BinRelation> childRelations = new HashSet<>();
    public Map<String, Object> feats = new HashMap<>();

    public Entity() {

    }

    public Entity(T t) {
        this.add(t);
    }

    @Override
    public String toString() {
        if (suggestedName != null) {
            return suggestedName;
        } else if (this.get(0) instanceof DNode) {
            return this.stream().map(x -> ((DNode) x).getForm()).reduce((x1, x2) -> x1 + StringUtils.SPACE + x2).orElse(StringUtils.EMPTY);
        } else {
            return this.stream().map(Object::toString).reduce((x1, x2) -> x1 + StringUtils.SPACE + x2).orElse(StringUtils.EMPTY);
        }
    }
}
