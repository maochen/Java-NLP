package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;

import java.util.UUID;

/**
 * Created by Maochen on 10/15/15.
 */
public class BinRelation {
    public UUID id = UUID.randomUUID();

    private String rel = StringUtils.EMPTY;
    private Entity left = null;
    private Entity right = null;

    public String getRel() {
        return rel;
    }

    public void setRel(String rel) {
        if (rel == null) {
            throw new IllegalArgumentException("Relation is null.");
        }
        this.rel = rel.toUpperCase();
    }


    public Entity getLeft() {
        return left;
    }

    public void setLeft(Entity left) {
        this.left = left;
        if (left != null) {
            left.binRelations.add(this);
        }
    }

    public Entity getRight() {
        return right;
    }

    public void setRight(Entity right) {
        this.right = right;
        if (right != null) {
            right.binRelations.add(this);
        }
    }

    @Override
    public String toString() {
        return "(" + rel + StringUtils.SPACE + left + StringUtils.SPACE + right + ") => " + id;
    }
}
