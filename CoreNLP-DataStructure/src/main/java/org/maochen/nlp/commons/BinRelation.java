package org.maochen.nlp.commons;

import org.apache.commons.lang3.StringUtils;

/**
 * Created by Maochen on 10/15/15.
 */
public class BinRelation<T> extends TupleRelation<T> {
    private Entity<T> left = null;
    private Entity<T> right = null;

    public Entity getLeft() {
        return left;
    }

    public void setLeft(Entity<T> left) {
        this.left = left;
        if (left != null) {
            left.binRelations.add(this);
        }
    }

    public Entity getRight() {
        return right;
    }

    public void setRight(Entity<T> right) {
        this.right = right;
        if (right != null) {
            right.binRelations.add(this);
        }
    }

    @Override
    public String toString() {
        return super.getRelType() + " = (" + super.getRel() + StringUtils.SPACE + left + StringUtils.SPACE + right + ") => " + id;
    }

//    public static void main(String[] args) {
//        BinRelation binRelation = new BinRelation();
//        binRelation.setRel("like");
//        binRelation.setRelType("VP");
//        binRelation.left = new Entity<>("Mary");
//        binRelation.right = new Entity<>("Tom");
//        System.out.println(binRelation);
//    }
}
