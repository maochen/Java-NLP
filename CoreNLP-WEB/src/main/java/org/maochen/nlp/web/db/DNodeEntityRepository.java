package org.maochen.nlp.web.db;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

/**
 * Created by mguan on 1/4/17.
 */
@Repository
public interface DNodeEntityRepository extends CrudRepository<DNodeEntity, UUID> {
}
