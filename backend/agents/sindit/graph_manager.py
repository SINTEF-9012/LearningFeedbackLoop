"""
Neo4j machine asset graph management.

Provides methods to initialize the graph schema, query machines and their
status, record status events, and maintain the asset graph lifecycle.

This module bridges the machine-centric schema (defined in schema.py) with
Neo4j operations using the neo4j driver or similar client.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from neo4j import Driver, AsyncDriver, AsyncSession, Session
except ImportError:
    Driver = None  # type: ignore
    AsyncDriver = None  # type: ignore

from .schema import (
    StatusCode,
    StatusSeverity,
    Machine,
    Status,
    Description,
    StatusEvent,
    ProductionLine,
    CYPHER_SCHEMA_DDL,
    CYPHER_SEED_DATA,
    QUERY_TEMPLATES,
)

logger = logging.getLogger(__name__)


class MachineAssetGraph:
    """Manages the Neo4j machine asset graph.
    
    Provides:
      - Schema initialization (constraints, indexes)
      - Machine CRUD operations
      - Status history tracking
      - Query templates for common operations
    """

    def __init__(self, driver: Optional[Any] = None):
        """Initialize with an optional Neo4j driver.
        
        Parameters
        ----------
        driver:
            Neo4j driver instance. If None, operations return stub/placeholder data.
        """
        self.driver = driver
        self._initialized = False

    # ====================================================================
    # Schema & Initialization
    # ====================================================================

    async def initialize_schema(self) -> bool:
        """Create constraints and indexes for the machine asset graph.
        
        Should be called once at startup.
        
        Returns
        -------
        True if schema initialization succeeded, False otherwise.
        """
        if not self.driver:
            logger.warning("MachineAssetGraph: No driver configured. Skipping schema init.")
            self._initialized = True
            return True

        try:
            # Split the DDL into individual statements and execute each
            statements = [s.strip() for s in CYPHER_SCHEMA_DDL.split(";") if s.strip()]
            async with self.driver.session() as session:
                for stmt in statements:
                    await session.run(stmt)
            self._initialized = True
            logger.info("Machine asset graph schema initialized")
            return True
        except Exception as exc:
            logger.error("Failed to initialize machine asset graph schema: %s", exc)
            return False

    async def seed_sample_data(self) -> bool:
        """Load sample machine and status data into the graph.
        
        Useful for development and testing.
        
        Returns
        -------
        True if seeding succeeded, False otherwise.
        """
        if not self.driver:
            logger.warning("MachineAssetGraph: No driver configured. Skipping seed.")
            return True

        try:
            # Split seed data into individual statements
            statements = [s.strip() for s in CYPHER_SEED_DATA.split(";") if s.strip()]
            async with self.driver.session() as session:
                for stmt in statements:
                    await session.run(stmt)
            logger.info("Machine asset graph seeded with sample data")
            return True
        except Exception as exc:
            logger.error("Failed to seed machine asset graph: %s", exc)
            return False

    async def ensure_status_catalog(self) -> bool:
        """Ensure canonical status nodes exist for CURRENT_STATUS relationships."""
        if not self.driver:
            return True
        try:
            rows = [
                (StatusCode.ACTIVE.value, "INFO", "Running"),
                (StatusCode.IDLE.value, "INFO", "Idle"),
                (StatusCode.STOPPED.value, "WARNING", "Stopped"),
                (StatusCode.MAINTENANCE.value, "INFO", "Maintenance"),
                (StatusCode.FAULT.value, "CRITICAL", "Fault"),
            ]
            query = """
                UNWIND $rows AS row
                MERGE (s:Status {code: row[0]})
                SET s.severity = row[1], s.label = row[2]
            """
            async with self.driver.session() as session:
                await session.run(query, rows=rows)
            return True
        except Exception as exc:
            logger.error("Failed to ensure status catalog: %s", exc)
            return False

    async def upsert_machine(
        self,
        *,
        machine_id: str,
        name: str,
        machine_type: str,
        vendor: Optional[str] = None,
        location: Optional[str] = None,
        active: bool = True,
        status: StatusCode = StatusCode.IDLE,
        status_reason: Optional[str] = None,
        status_source: str = "api",
        metadata_json: str = "{}",
    ) -> Optional[Dict[str, Any]]:
        """Create or update a machine, set CURRENT_STATUS, and append status event."""
        if not self.driver:
            return {
                "machine_id": machine_id,
                "name": name,
                "status_code": status.value,
            }

        try:
            await self.ensure_status_catalog()
            async with self.driver.session() as session:
                query = """
                    MERGE (m:Machine {machine_id: $machine_id})
                    ON CREATE SET m.created_at = $updated_at
                    SET m.name = $name,
                        m.type = $type,
                        m.vendor = $vendor,
                        m.location = $location,
                        m.active = $active,
                        m.updated_at = $updated_at,
                        m.sindit_metadata_json = $metadata_json
                    WITH m
                    OPTIONAL MATCH (m)-[old:CURRENT_STATUS]->(prev:Status)
                    DELETE old
                    WITH m, prev
                    MATCH (s:Status {code: $status_code})
                    MERGE (m)-[:CURRENT_STATUS]->(s)
                    WITH m, s, prev
                    CREATE (e:StatusEvent {
                        timestamp: $updated_at,
                        from_status: COALESCE(prev.code, $status_code),
                        to_status: $status_code,
                        reason: $reason,
                        source: $source
                    })
                    MERGE (m)-[:HAS_STATUS_EVENT]->(e)
                    RETURN m.machine_id AS machine_id, m.name AS name, s.code AS status_code
                """
                result = await session.run(
                    query,
                    machine_id=machine_id,
                    name=name,
                    type=machine_type,
                    vendor=vendor or "",
                    location=location or "",
                    active=active,
                    updated_at=datetime.utcnow().isoformat(),
                    status_code=status.value,
                    reason=status_reason or "",
                    source=status_source,
                    metadata_json=metadata_json,
                )
                row = await result.single()
                if row:
                    return {
                        "machine_id": row["machine_id"],
                        "name": row["name"],
                        "status_code": row["status_code"],
                    }
        except Exception as exc:
            logger.error("Failed to upsert machine %s: %s", machine_id, exc)
        return None

    # ====================================================================
    # Machine Operations
    # ====================================================================

    async def create_machine(
        self,
        machine_id: str,
        name: str,
        machine_type: str,
        vendor: Optional[str] = None,
        location: Optional[str] = None,
        active: bool = True,
        description_text: Optional[str] = None,
        initial_status: StatusCode = StatusCode.STOPPED,
    ) -> Optional[Dict[str, Any]]:
        """Create a new machine asset with optional initial status and description.
        
        Parameters
        ----------
        machine_id:
            Unique machine identifier (e.g., "MACHINE_A1")
        name:
            Display name
        machine_type:
            Equipment type
        vendor:
            Manufacturing vendor
        location:
            Physical location
        active:
            Is machine in service?
        description_text:
            Optional human-readable description
        initial_status:
            Initial status code (default: STOPPED)
        
        Returns
        -------
        Dict with machine_id, name, status_code, or None on error.
        """
        if not self.driver:
            return {
                "machine_id": machine_id,
                "name": name,
                "status_code": initial_status.value,
            }

        try:
            async with self.driver.session() as session:
                # Create machine node
                query = """
                    CREATE (m:Machine {
                        machine_id: $machine_id,
                        name: $name,
                        type: $type,
                        vendor: $vendor,
                        location: $location,
                        active: $active,
                        updated_at: $updated_at
                    })
                    OPTIONAL MATCH (s:Status {code: $status_code})
                    CREATE (m)-[:CURRENT_STATUS]->(s)
                    RETURN m.machine_id, m.name, s.code
                """
                result = await session.run(
                    query,
                    machine_id=machine_id,
                    name=name,
                    type=machine_type,
                    vendor=vendor or "",
                    location=location or "",
                    active=active,
                    updated_at=datetime.utcnow().isoformat(),
                    status_code=initial_status.value,
                )
                record = await result.single()
                if record:
                    logger.info("Created machine %s (%s)", machine_id, name)
                    
                    # Add description if provided
                    if description_text:
                        await self.add_machine_description(machine_id, description_text)
                    
                    return {
                        "machine_id": record["m.machine_id"],
                        "name": record["m.name"],
                        "status_code": record.get("s.code"),
                    }
        except Exception as exc:
            logger.error("Failed to create machine %s: %s", machine_id, exc)
        return None

    async def get_machine(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Get machine with current status and description.
        
        Returns
        -------
        Dict with machine properties, current_status, description, updated_at, or None.
        """
        if not self.driver:
            return None

        try:
            async with self.driver.session() as session:
                query = QUERY_TEMPLATES["get_machine_with_status"]
                result = await session.run(query, machine_id=machine_id)
                record = await result.single()
                if record:
                    m = record.get("m")
                    s = record.get("s")
                    d = record.get("d")
                    return {
                        "machine_id": m.get("machine_id") if m else None,
                        "name": m.get("name") if m else None,
                        "type": m.get("type") if m else None,
                        "vendor": m.get("vendor") if m else None,
                        "location": m.get("location") if m else None,
                        "active": m.get("active", True) if m else None,
                        "current_status": s.get("code") if s else None,
                        "description": d.get("text") if d else None,
                        "updated_at": m.get("updated_at") if m else None,
                    }
        except Exception as exc:
            logger.debug("Failed to get machine %s: %s", machine_id, exc)
        return None

    async def list_machines(self) -> List[Dict[str, Any]]:
        """List all active machines with current status.
        
        Returns
        -------
        List of machine dicts with current status, or empty list on error.
        """
        if not self.driver:
            return []

        try:
            async with self.driver.session() as session:
                query = QUERY_TEMPLATES["list_active_machines"]
                result = await session.run(query)
                records = await result.fetch(None)  # Fetch all
                machines = []
                for record in records:
                    machines.append({
                        "machine_id": record[0],
                        "name": record[1],
                        "status": record[2],
                        "status_label": record[3],
                        "description": record[4],
                    })
                return machines
        except Exception as exc:
            logger.error("Failed to list machines: %s", exc)
        return []

    # ====================================================================
    # Status Management
    # ====================================================================

    async def update_machine_status(
        self,
        machine_id: str,
        new_status: StatusCode,
        reason: Optional[str] = None,
        source: str = "api",
    ) -> bool:
        """Update machine status and record a status event.
        
        Parameters
        ----------
        machine_id:
            Target machine
        new_status:
            New status code
        reason:
            Why the status changed
        source:
            Source of the change (e.g., "operator", "model", "sindit")
        
        Returns
        -------
        True if update succeeded, False otherwise.
        """
        if not self.driver:
            logger.info("Would update machine %s status to %s", machine_id, new_status.value)
            return True

        try:
            async with self.driver.session() as session:
                # Get current status
                current = await self.get_machine_current_status(machine_id)
                
                # Update status
                query = QUERY_TEMPLATES["update_machine_status"]
                await session.run(
                    query,
                    machine_id=machine_id,
                    status_code=new_status.value,
                    updated_at=datetime.utcnow().isoformat(),
                )
                
                # Record status event
                if current:
                    await self.record_status_event(
                        machine_id,
                        from_status=StatusCode(current),
                        to_status=new_status,
                        reason=reason,
                        source=source,
                    )
                
                logger.info(
                    "Updated machine %s status to %s (reason: %s, source: %s)",
                    machine_id, new_status.value, reason or "none", source,
                )
                return True
        except Exception as exc:
            logger.error("Failed to update machine %s status: %s", machine_id, exc)
        return False

    async def get_machine_current_status(self, machine_id: str) -> Optional[str]:
        """Get the current status code for a machine.
        
        Returns
        -------
        Status code string (e.g., "ACTIVE") or None if machine not found.
        """
        machine = await self.get_machine(machine_id)
        return machine.get("current_status") if machine else None

    # ====================================================================
    # Status Events (History)
    # ====================================================================

    async def record_status_event(
        self,
        machine_id: str,
        from_status: StatusCode,
        to_status: StatusCode,
        reason: Optional[str] = None,
        source: str = "api",
    ) -> bool:
        """Record a status transition event in the history.
        
        Returns
        -------
        True if event recorded, False on error.
        """
        if not self.driver:
            logger.info(
                "Would record event for %s: %s → %s",
                machine_id, from_status.value, to_status.value,
            )
            return True

        try:
            async with self.driver.session() as session:
                query = QUERY_TEMPLATES["record_status_event"]
                await session.run(
                    query,
                    machine_id=machine_id,
                    timestamp=datetime.utcnow().isoformat(),
                    from_status=from_status.value,
                    to_status=to_status.value,
                    reason=reason or "",
                    source=source,
                )
                logger.debug("Recorded status event for %s", machine_id)
                return True
        except Exception as exc:
            logger.error("Failed to record status event for %s: %s", machine_id, exc)
        return False

    async def get_status_history(
        self, machine_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get status change history for a machine.
        
        Returns
        -------
        List of status events (newest first), ordered by timestamp DESC.
        """
        if not self.driver:
            return []

        try:
            async with self.driver.session() as session:
                query = QUERY_TEMPLATES["get_machine_status_history"]
                result = await session.run(query, machine_id=machine_id, limit=limit)
                records = await result.fetch(None)
                events = []
                for record in records:
                    events.append({
                        "timestamp": record[0],
                        "from_status": record[1],
                        "to_status": record[2],
                        "reason": record[3],
                        "source": record[4],
                    })
                return events
        except Exception as exc:
            logger.error("Failed to get status history for %s: %s", machine_id, exc)
        return []

    # ====================================================================
    # Descriptions & Metadata
    # ====================================================================

    async def add_machine_description(
        self,
        machine_id: str,
        description_text: str,
        lang: str = "en",
        source: str = "manual",
    ) -> bool:
        """Add or update a machine description.
        
        Returns
        -------
        True if operation succeeded, False otherwise.
        """
        if not self.driver:
            return True

        try:
            async with self.driver.session() as session:
                query = """
                    MATCH (m:Machine {machine_id: $machine_id})
                    OPTIONAL MATCH (m)-[r:HAS_DESCRIPTION]->()
                    DELETE r
                    WITH m
                    CREATE (d:Description {
                        text: $text,
                        lang: $lang,
                        source: $source
                    })
                    CREATE (m)-[:HAS_DESCRIPTION]->(d)
                    RETURN d
                """
                result = await session.run(
                    query,
                    machine_id=machine_id,
                    text=description_text,
                    lang=lang,
                    source=source,
                )
                record = await result.single()
                if record:
                    logger.debug("Added description for machine %s", machine_id)
                    return True
        except Exception as exc:
            logger.error("Failed to add description for %s: %s", machine_id, exc)
        return False

    # ====================================================================
    # Production Lines
    # ====================================================================

    async def create_production_line(
        self,
        line_id: str,
        name: str,
        location: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a production line asset.
        
        Returns
        -------
        Dict with line_id, name, location, or None on error.
        """
        if not self.driver:
            return {"line_id": line_id, "name": name, "location": location}

        try:
            async with self.driver.session() as session:
                query = """
                    MERGE (l:ProductionLine {line_id: $line_id})
                    SET l.name = $name,
                        l.location = $location
                    RETURN l.line_id, l.name
                """
                result = await session.run(
                    query,
                    line_id=line_id,
                    name=name,
                    location=location or "",
                )
                record = await result.single()
                if record:
                    logger.info("Created production line %s", line_id)
                    return {
                        "line_id": record[0],
                        "name": record[1],
                        "location": location,
                    }
        except Exception as exc:
            logger.error("Failed to create production line %s: %s", line_id, exc)
        return None

    async def add_machine_to_line(
        self, line_id: str, machine_id: str
    ) -> bool:
        """Add a machine to a production line.
        
        Returns
        -------
        True if operation succeeded, False otherwise.
        """
        if not self.driver:
            return True

        try:
            async with self.driver.session() as session:
                query = """
                    MATCH (line:ProductionLine {line_id: $line_id}),
                          (m:Machine {machine_id: $machine_id})
                    MERGE (line)-[:HAS_MACHINE]->(m)
                    RETURN line, m
                """
                result = await session.run(
                    query, line_id=line_id, machine_id=machine_id
                )
                record = await result.single()
                if record:
                    logger.debug("Added machine %s to line %s", machine_id, line_id)
                    return True
        except Exception as exc:
            logger.error(
                "Failed to add machine %s to line %s: %s", machine_id, line_id, exc
            )
        return False

    async def get_production_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        """Get production line with all its machines and their statuses.
        
        Returns
        -------
        Dict with line info and list of machines, or None.
        """
        if not self.driver:
            return None

        try:
            async with self.driver.session() as session:
                query = QUERY_TEMPLATES["get_production_line"]
                result = await session.run(query, line_id=line_id)
                records = await result.fetch(None)
                if not records:
                    return None
                
                line_data = None
                machines = []
                for record in records:
                    line = record.get("line")
                    if not line_data:
                        line_data = {
                            "line_id": line.get("line_id"),
                            "name": line.get("name"),
                            "location": line.get("location"),
                        }
                    m = record.get("m")
                    s = record.get("s")
                    machines.append({
                        "machine_id": m.get("machine_id"),
                        "name": m.get("name"),
                        "status": s.get("code") if s else None,
                    })
                
                if line_data:
                    line_data["machines"] = machines
                return line_data
        except Exception as exc:
            logger.error("Failed to get production line %s: %s", line_id, exc)
        return None
