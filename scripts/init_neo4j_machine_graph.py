#!/usr/bin/env python3
"""
Neo4j Machine Asset Graph Initialization Script

This script initializes the Neo4j database with the machine-centric schema
and optionally loads sample data.

Usage:
    python3 scripts/init_neo4j_machine_graph.py [--seed] [--uri neo4j://...] [--username neo4j] [--password changeme]

Environment variables (overrides CLI args):
    NEO4J_URI:       Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USERNAME:  Neo4j username (default: neo4j)
    NEO4J_PASSWORD:  Neo4j password (default: changeme)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Setup path to import backend modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="Initialize Neo4j machine asset graph schema"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Load sample machine and status data after schema init",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Neo4j connection URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password (default: changeme)",
    )
    args = parser.parse_args()

    # Import here to avoid import errors if neo4j not installed
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        logger.error(
            "neo4j package not installed. Install with: pip install neo4j"
        )
        return 1

    # Get connection params from env or CLI args
    import os
    uri = args.uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = args.username or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = args.password or os.environ.get("NEO4J_PASSWORD", "changeme")

    logger.info(f"Connecting to Neo4j at {uri}...")

    try:
        # Create driver
        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

        # Test connection
        async with driver.session() as session:
            result = await session.run("RETURN 1")
            await result.single()
        logger.info("Connected to Neo4j successfully")

        # Initialize graph manager
        from backend.agents.sindit.graph_manager import MachineAssetGraph

        graph = MachineAssetGraph(driver)

        # Initialize schema
        logger.info("Initializing machine asset graph schema...")
        schema_ok = await graph.initialize_schema()
        if not schema_ok:
            logger.error("Failed to initialize schema")
            return 2

        logger.info("Schema initialized successfully")

        # Load seed data if requested
        if args.seed:
            logger.info("Loading sample machine and status data...")
            seed_ok = await graph.seed_sample_data()
            if not seed_ok:
                logger.error("Failed to seed data")
                return 3
            logger.info("Sample data loaded successfully")

        # Print schema info
        logger.info("Machine asset graph ready")
        logger.info("Schema includes:")
        logger.info("  - Machine nodes (with machine_id, name, type, vendor, location, active, updated_at)")
        logger.info("  - Status nodes (ACTIVE, IDLE, STOPPED, MAINTENANCE, FAULT)")
        logger.info("  - Description nodes (for machine documentation)")
        logger.info("  - StatusEvent nodes (historical status transitions)")
        logger.info("  - ProductionLine nodes (for grouping/hierarchy)")
        logger.info("Relationships:")
        logger.info("  - (Machine)-[:CURRENT_STATUS]->(Status)")
        logger.info("  - (Machine)-[:HAS_DESCRIPTION]->(Description)")
        logger.info("  - (Machine)-[:HAS_STATUS_EVENT]->(StatusEvent)")
        logger.info("  - (ProductionLine)-[:HAS_MACHINE]->(Machine)")

        await driver.close()
        return 0

    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
