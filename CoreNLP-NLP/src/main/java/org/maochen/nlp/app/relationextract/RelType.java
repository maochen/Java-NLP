package org.maochen.nlp.app.relationextract;

/**
 * Created by Maochen on 11/26/15.
 */
public enum RelType {
    WILDCARD("wildcard"),
    ISA("isa");

    String val;

    RelType(String val) {
        this.val = val;
    }
}
