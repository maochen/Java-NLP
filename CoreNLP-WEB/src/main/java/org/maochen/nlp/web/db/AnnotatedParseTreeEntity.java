package org.maochen.nlp.web.db;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.LangTools;

import java.util.Date;
import java.util.UUID;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import javax.validation.constraints.NotNull;

/**
 * Created by mguan on 12/30/16.
 */
@Entity
@Table(name = "annotated_parse_tree")
public class AnnotatedParseTreeEntity {

  @Id
  public UUID id;

  @NotNull
  public DTree expectedParseTree;

  @Column(unique = true)
  public String sentence;

  @NotNull
  public String author;

  public Date date;

  public String comment = StringUtils.EMPTY;

  public AnnotatedParseTreeEntity() {

  }

  public AnnotatedParseTreeEntity(DTree expectedParseTree, String sentence, String author) {
    this.expectedParseTree = expectedParseTree;
    this.sentence = sentence;
    this.author = author;
    this.date = new Date();
    this.id = expectedParseTree.getUUID();
  }


  public AnnotatedParseTreeEntity(String expectedCoNLLString, String sentence, String author) {
    this(LangTools.getDTreeFromCoNLLXString(expectedCoNLLString), sentence, author);
  }

  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append(this.id.toString()).append("\t");
    stringBuilder.append(this.sentence).append("\t");
    stringBuilder.append(this.expectedParseTree).append("\t");
    stringBuilder.append(this.author).append("\t");
    stringBuilder.append(this.date).append("\t");
    stringBuilder.append(this.comment);
    return stringBuilder.toString();
  }
}
