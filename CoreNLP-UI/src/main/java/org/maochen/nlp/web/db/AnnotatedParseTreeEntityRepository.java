package org.maochen.nlp.web.db;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

/**
 * Created by mguan on 12/30/16.
 */
@Repository
public interface AnnotatedParseTreeEntityRepository extends CrudRepository<AnnotatedParseTreeEntity, UUID> {
  List<AnnotatedParseTreeEntity> findBySentence(String sentence);
}
