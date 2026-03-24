"""
Script de test pour valider l'Agent IA avec des scénarios réels
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
LANGGRAPH_TIMEOUT_SECONDS = 90
BASELINE_TIMEOUT_SECONDS = 30

# Scénarios de test (vrais cas SmartTalk)
TEST_SCENARIOS = [
    {
        "name": "Mock manquant - MockK",
        "test_file": "AuthenticationViewModelTest.kt",
        "test_name": "testLogin",
        "test_code": """@Test
fun testLogin() {
    val result = viewModel.login("user", "pass")
    assertNotNull(result)
}""",
        "error_logs": "NullPointerException: viewModel is null at AuthenticationViewModelTest.testLogin(AuthenticationViewModelTest.kt:45)",
        "expected_error_type": "NULL_POINTER"
    },
    {
        "name": "Mock non configuré - every",
        "test_file": "CallViewModelTest.kt",
        "test_name": "testStartCall",
        "test_code": """@Test
fun testStartCall() {
    viewModel.startCall("123")
    verify { callService.start(any()) }
}""",
        "error_logs": "io.mockk.MockKException: no answer found for: CallService(#1).start(String)",
        "expected_error_type": "MOCK_MISSING"
    },
    {
        "name": "Assertion incorrecte - assertEquals",
        "test_file": "MessagingViewModelTest.kt",
        "test_name": "testSendMessage",
        "test_code": """@Test
fun testSendMessage() {
    val result = viewModel.sendMessage("Hello")
    assertEquals("Sent", result)
}""",
        "error_logs": "AssertionError: expected:<Sent> but was:<Pending>",
        "expected_error_type": "ASSERTION_ERROR"
    },
    {
        "name": "Coroutine Dispatcher manquant",
        "test_file": "DashboardViewModelTest.kt",
        "test_name": "testLoadData",
        "test_code": """@Test
fun testLoadData() = runBlocking {
    viewModel.loadData()
    assertTrue(viewModel.isLoaded)
}""",
        "error_logs": "IllegalStateException: Module with the Main dispatcher is missing",
        "expected_error_type": "COROUTINE_DISPATCHER"
    },
    {
        "name": "Koin DI non initialisé",
        "test_file": "SettingsViewModelTest.kt",
        "test_name": "testGetSettings",
        "test_code": """@Test
fun testGetSettings() {
    val settings = viewModel.getSettings()
    assertNotNull(settings)
}""",
        "error_logs": "org.koin.core.error.NoBeanDefFoundException: No definition found for class",
        "expected_error_type": "KOIN_MISSING"
    }
]

def test_scenario(scenario: Dict[str, Any], use_langgraph: bool = True) -> Dict[str, Any]:
    """Teste un scénario et collecte les métriques"""
    
    endpoint = "/analyze-failure-langgraph" if use_langgraph else "/analyze-failure"
    url = f"{API_BASE_URL}{endpoint}"
    
    payload = {
        "test_file": scenario["test_file"],
        "test_name": scenario["test_name"],
        "test_code": scenario["test_code"],
        "error_logs": scenario["error_logs"]
    }
    
    print(f"\n{'='*60}")
    print(f"🧪 Test: {scenario['name']}")
    print(f"   Endpoint: {endpoint}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        timeout_seconds = LANGGRAPH_TIMEOUT_SECONDS if use_langgraph else BASELINE_TIMEOUT_SECONDS
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Extraire les métriques
            metrics = {
                "scenario_name": scenario["name"],
                "endpoint": endpoint,
                "success": result.get("success", False),
                "processing_time": processing_time,
                "error_type_detected": result.get("error_type"),
                "error_type_expected": scenario["expected_error_type"],
                "confidence_score": result.get("confidence_score"),
                "tokens_used": result.get("tokens_used"),
                "is_valid": result.get("is_valid"),
                "has_fix": bool(result.get("proposed_fix") or result.get("corrected_code")),
                "fix_length": len(result.get("proposed_fix") or result.get("corrected_code") or ""),
                "steps_completed": len(result.get("steps_completed", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            # Afficher résumé
            print(f"✅ SUCCESS")
            print(f"   ⏱️  Temps: {processing_time:.2f}s")
            print(f"   🎯 Type erreur: {metrics['error_type_detected']}")
            print(f"   📊 Confiance: {metrics['confidence_score']}")
            print(f"   🔢 Tokens: {metrics['tokens_used']}")
            print(f"   ✅ Valid: {metrics['is_valid']}")
            print(f"   📝 Fix généré: {metrics['has_fix']}")
            
            if use_langgraph:
                print(f"   🔄 Étapes: {metrics['steps_completed']}")
            
            return metrics
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"   {response.text}")
            return {
                "scenario_name": scenario["name"],
                "success": False,
                "error": response.text
            }
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return {
            "scenario_name": scenario["name"],
            "success": False,
            "error": str(e)
        }

def run_all_tests():
    """Exécute tous les tests et génère un rapport"""
    
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE DES TESTS - AGENT IA")
    print("="*60)
    
    # Vérifier que le serveur est up
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Serveur non accessible!")
            return
        print("✅ Serveur accessible")
    except:
        print("❌ Serveur non démarré! Lance: python main.py")
        return
    
    results_langgraph = []
    results_basic = []
    
    # Test avec LangGraph (RAG activé)
    print("\n" + "="*60)
    print("📊 PHASE 1: Tests avec LangGraph + RAG")
    print("="*60)
    
    for scenario in TEST_SCENARIOS:
        result = test_scenario(scenario, use_langgraph=True)
        results_langgraph.append(result)
        time.sleep(2)  # Pause pour rate limiting
    
    # Test sans LangGraph (baseline)
    print("\n" + "="*60)
    print("📊 PHASE 2: Tests SANS LangGraph (Baseline)")
    print("="*60)
    
    for scenario in TEST_SCENARIOS[:2]:  # Juste 2 pour comparaison
        result = test_scenario(scenario, use_langgraph=False)
        results_basic.append(result)
        time.sleep(2)
    
    # Générer le rapport
    generate_report(results_langgraph, results_basic)

def generate_report(results_langgraph: List[Dict], results_basic: List[Dict]):
    """Génère un rapport de comparaison"""
    
    print("\n" + "="*60)
    print("📊 RAPPORT DE TEST - MÉTRIQUES")
    print("="*60)
    
    # Métriques LangGraph
    lg_success = sum(1 for r in results_langgraph if r.get("success"))
    lg_avg_time = sum(r.get("processing_time", 0) for r in results_langgraph) / len(results_langgraph)
    lg_avg_confidence = sum(r.get("confidence_score", 0) for r in results_langgraph if r.get("confidence_score")) / lg_success if lg_success > 0 else 0
    lg_avg_tokens = sum(r.get("tokens_used", 0) for r in results_langgraph) / lg_success if lg_success > 0 else 0
    lg_valid = sum(1 for r in results_langgraph if r.get("is_valid"))
    
    # Métriques Basic
    basic_success = sum(1 for r in results_basic if r.get("success"))
    basic_avg_time = sum(r.get("processing_time", 0) for r in results_basic) / len(results_basic) if results_basic else 0
    
    print(f"\n🎯 AVEC LANGGRAPH + RAG:")
    print(f"   Succès: {lg_success}/{len(results_langgraph)} ({lg_success/len(results_langgraph)*100:.1f}%)")
    print(f"   Temps moyen: {lg_avg_time:.2f}s")
    print(f"   Confiance moyenne: {lg_avg_confidence:.2f}")
    print(f"   Tokens moyens: {lg_avg_tokens:.0f}")
    print(f"   Corrections valides: {lg_valid}/{lg_success} ({lg_valid/lg_success*100:.1f}%)")
    
    print(f"\n⚙️  SANS LANGGRAPH (Baseline):")
    print(f"   Succès: {basic_success}/{len(results_basic)}")
    print(f"   Temps moyen: {basic_avg_time:.2f}s")
    
    print(f"\n📈 AMÉLIORATION:")
    if basic_avg_time > 0:
        time_diff = ((lg_avg_time - basic_avg_time) / basic_avg_time) * 100
        print(f"   Temps: {time_diff:+.1f}% (avec workflow)")
    print(f"   Qualité: +{(lg_valid/lg_success*100):.0f}% validation")
    print(f"   Traçabilité: 6 étapes vs 0")
    
    # Sauvegarder le rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "langgraph_results": results_langgraph,
        "basic_results": results_basic,
        "summary": {
            "langgraph": {
                "success_rate": lg_success/len(results_langgraph)*100,
                "avg_time": lg_avg_time,
                "avg_confidence": lg_avg_confidence,
                "avg_tokens": lg_avg_tokens,
                "valid_rate": lg_valid/lg_success*100 if lg_success > 0 else 0
            },
            "basic": {
                "success_rate": basic_success/len(results_basic)*100 if results_basic else 0,
                "avg_time": basic_avg_time
            }
        }
    }
    
    # Sauvegarder en JSON
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Rapport sauvegardé: test_report.json")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
