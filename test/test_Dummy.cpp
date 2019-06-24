#include <boost/test/unit_test.hpp>
#include <motion_planning/Dummy.hpp>

using namespace motion_planning;

BOOST_AUTO_TEST_CASE(it_should_not_crash_when_welcome_is_called)
{
    motion_planning::DummyClass dummy;
    dummy.welcome();
}
