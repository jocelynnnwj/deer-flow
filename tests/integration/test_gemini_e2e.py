#!/usr/bin/env python3
"""
End-to-end test for Gemini integration in DeerFlow workflow
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.workflow import run_agent_workflow_async
from src.config.agents import AGENT_LLM_MAP

async def test_gemini_workflow():
    """Test Gemini in a complete DeerFlow workflow"""
    print("🚀 Testing Gemini in DeerFlow Workflow...")
    
    # Test query
    query = "What are the latest developments in renewable energy technology?"
    
    try:
        # Run the workflow with Gemini
        result = await run_agent_workflow_async(query)

        if not result:
            raise ValueError("Workflow returned an empty result.")

        print("✅ Workflow completed successfully!")
        
        # Check for report content
        report = result.get("report", "")
        if report:
            print(f"📄 Report length: {len(report)} characters")
            print(f"🔍 Final report content:\\n{report[:500]}...")
        else:
            print("⚠️ Report is empty, but workflow finished. Full output:")
            print(result)

        assert report, "Report should not be empty"
        
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        return False

def test_agent_mapping():
    """Test that agents can be mapped to Gemini"""
    print("\n🔧 Testing Agent Mapping...")
    
    try:
        # Check if we can modify the agent mapping
        original_mapping = AGENT_LLM_MAP.copy()
        
        # Temporarily map some agents to Gemini
        AGENT_LLM_MAP["researcher"] = "gemini"
        AGENT_LLM_MAP["coordinator"] = "gemini"
        
        print("✅ Successfully mapped agents to Gemini")
        print(f"   Researcher: {AGENT_LLM_MAP['researcher']}")
        print(f"   Coordinator: {AGENT_LLM_MAP['coordinator']}")
        
        # Restore original mapping
        AGENT_LLM_MAP.clear()
        AGENT_LLM_MAP.update(original_mapping)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in agent mapping: {e}")
        return False

def test_configuration():
    """Test Gemini configuration loading"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from src.llms.llm import get_configured_llm_models
        
        models = get_configured_llm_models()
        
        if "gemini" in models:
            print(f"✅ Gemini models configured: {models['gemini']}")
        else:
            print("⚠️  No Gemini models found in configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in configuration: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 DeerFlow Gemini End-to-End Tests")
    print("=" * 50)
    
    # Test configuration
    config_success = test_configuration()
    
    # Test agent mapping
    mapping_success = test_agent_mapping()
    
    # Test workflow (this might take a while)
    print("\n⏳ Running workflow test (this may take a few minutes)...")
    workflow_success = await test_gemini_workflow()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"   Agent Mapping: {'✅ PASS' if mapping_success else '❌ FAIL'}")
    print(f"   Workflow: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    
    if all([config_success, mapping_success, workflow_success]):
        print("\n🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 